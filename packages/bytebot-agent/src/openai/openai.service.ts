import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI, { APIUserAbortError } from 'openai';
import {
  MessageContentBlock,
  MessageContentType,
  TextContentBlock,
  ToolUseContentBlock,
  ToolResultContentBlock,
  ThinkingContentBlock,
  isUserActionContentBlock,
  isComputerToolUseContentBlock,
  isImageContentBlock,
} from '@bytebot/shared';
import { DEFAULT_MODEL } from './openai.constants';
import { Message, Role } from '@prisma/client';
import { openaiTools } from './openai.tools';
import {
  BytebotAgentService,
  BytebotAgentInterrupt,
  BytebotAgentResponse,
} from '../agent/agent.types';

@Injectable()
export class OpenAIService implements BytebotAgentService {
  private readonly openai: OpenAI;
  private readonly logger = new Logger(OpenAIService.name);

  constructor(private readonly configService: ConfigService) {
    const apiKey = this.configService.get<string>('OPENAI_API_KEY');

    if (!apiKey) {
      this.logger.warn(
        'OPENAI_API_KEY is not set. OpenAIService will not work properly.',
      );
    }

    // Support OpenRouter or any OpenAI-compatible base via env
    const baseURL =
      this.configService.get<string>('OPENAI_API_BASE') ||
      this.configService.get<string>('OPENAI_BASE_URL') ||
      this.configService.get<string>('OPENAI_API_HOST') ||
      undefined;

    const httpReferer = this.configService.get<string>('HTTP_REFERER');
    const xTitle = this.configService.get<string>('X_TITLE');

    this.openai = new OpenAI({
      apiKey: apiKey || 'dummy-key-for-initialization',
      baseURL,
      defaultHeaders: {
        ...(!!httpReferer ? { 'HTTP-Referer': httpReferer } : {}),
        ...(!!xTitle ? { 'X-Title': xTitle } : {}),
      },
    });
  }

  async generateMessage(
    systemPrompt: string,
    messages: Message[],
    model: string = DEFAULT_MODEL.name,
    useTools: boolean = true,
    signal?: AbortSignal,
  ): Promise<BytebotAgentResponse> {
    const isReasoning = model.startsWith('o');
    try {
      const openaiMessages = this.formatMessagesForOpenAI(messages);

      const maxTokens = 8192;
      const response = await this.openai.responses.create(
        {
          model,
          max_output_tokens: maxTokens,
          input: openaiMessages,
          instructions: systemPrompt,
          tools: useTools ? openaiTools : [],
          reasoning: isReasoning ? { effort: 'medium' } : null,
          store: false,
          include: isReasoning ? ['reasoning.encrypted_content'] : [],
        },
        { signal },
      );

      return {
        contentBlocks: this.formatOpenAIResponse(response.output),
        tokenUsage: {
          inputTokens: response.usage?.input_tokens || 0,
          outputTokens: response.usage?.output_tokens || 0,
          totalTokens: response.usage?.total_tokens || 0,
        },
      };
    } catch (error: any) {
      console.log('error', error);
      console.log('error name', error.name);

      // Fallback: if using OpenRouter (or non-Responses-compatible provider), try Chat Completions
      const base =
        this.configService.get<string>('OPENAI_API_BASE') ||
        this.configService.get<string>('OPENAI_BASE_URL') ||
        '';

      const likelyNeedsChatAPI =
        base.includes('openrouter.ai') ||
        (error?.status === 404) ||
        (typeof error?.message === 'string' && /responses/i.test(error.message));

      if (likelyNeedsChatAPI) {
        try {
          const chatMessages = this.formatMessagesForChat(messages, systemPrompt);

          const chatTools = useTools
            ? (openaiTools as any[]).map((t: any) => ({
                type: 'function',
                function: {
                  name: t.name,
                  description: t.description,
                  parameters: t.parameters,
                },
              }))
            : undefined;

          const completion = await this.openai.chat.completions.create(
            {
              model,
              messages: chatMessages as any,
              tools: chatTools as any,
              tool_choice: useTools ? 'auto' : undefined,
            },
            { signal },
          );

          const choice = completion.choices?.[0];
          const msg: any = choice?.message;

          const blocks: MessageContentBlock[] = [];

          // text/content
          if (msg?.content) {
            const finalText =
              typeof msg.content === 'string'
                ? msg.content
                : Array.isArray(msg.content)
                ? msg.content
                    .map((p: any) =>
                      typeof p === 'string' ? p : p?.text || '',
                    )
                    .filter(Boolean)
                    .join('\n')
                : '';
            if (finalText) {
              blocks.push({
                type: MessageContentType.Text,
                text: finalText,
              } as TextContentBlock);
            }
          }

          // tool calls
          if (Array.isArray(msg?.tool_calls)) {
            for (const tc of msg.tool_calls) {
              const name = tc?.function?.name;
              let args: any = {};
              try {
                args = tc?.function?.arguments
                  ? JSON.parse(tc.function.arguments)
                  : {};
              } catch {
                args = {};
              }
              blocks.push({
                type: MessageContentType.ToolUse,
                id: tc?.id || name || 'tool_call',
                name,
                input: args,
              } as ToolUseContentBlock);
            }
          }

          return {
            contentBlocks: blocks,
            tokenUsage: {
              inputTokens: (completion as any).usage?.prompt_tokens || 0,
              outputTokens: (completion as any).usage?.completion_tokens || 0,
              totalTokens: (completion as any).usage?.total_tokens || 0,
            },
          };
        } catch (fallbackErr: any) {
          this.logger.warn(
            `Chat Completions fallback failed: ${fallbackErr?.message || fallbackErr}`,
          );
        }
      }

      if (error instanceof APIUserAbortError) {
        this.logger.log('OpenAI API call aborted');
        throw new BytebotAgentInterrupt();
      }
      this.logger.error(
        `Error sending message to OpenAI: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  private formatMessagesForOpenAI(
    messages: Message[],
  ): OpenAI.Responses.ResponseInputItem[] {
    const openaiMessages: OpenAI.Responses.ResponseInputItem[] = [];

    for (const message of messages) {
      const messageContentBlocks = message.content as MessageContentBlock[];

      if (
        messageContentBlocks.every((block) => isUserActionContentBlock(block))
      ) {
        const userActionContentBlocks = messageContentBlocks.flatMap(
          (block) => block.content,
        );
        for (const block of userActionContentBlocks) {
          if (isComputerToolUseContentBlock(block)) {
            openaiMessages.push({
              type: 'message',
              role: 'user',
              content: [
                {
                  type: 'input_text',
                  text: `User performed action: ${block.name}\n${JSON.stringify(block.input, null, 2)}`,
                },
              ],
            });
          } else if (isImageContentBlock(block)) {
            openaiMessages.push({
              role: 'user',
              type: 'message',
              content: [
                {
                  type: 'input_image',
                  detail: 'high',
                  image_url: `data:${block.source.media_type};base64,${block.source.data}`,
                },
              ],
            } as OpenAI.Responses.ResponseInputItem.Message);
          }
        }
      } else {
        // Convert content blocks to OpenAI format
        for (const block of messageContentBlocks) {
          switch (block.type) {
            case MessageContentType.Text: {
              if (message.role === Role.USER) {
                openaiMessages.push({
                  type: 'message',
                  role: 'user',
                  content: [
                    {
                      type: 'input_text',
                      text: block.text,
                    },
                  ],
                } as OpenAI.Responses.ResponseInputItem.Message);
              } else {
                openaiMessages.push({
                  type: 'message',
                  role: 'assistant',
                  content: [
                    {
                      type: 'output_text',
                      text: block.text,
                    },
                  ],
                } as OpenAI.Responses.ResponseOutputMessage);
              }
              break;
            }
            case MessageContentType.ToolUse:
              // For assistant messages with tool use, convert to function call
              if (message.role === Role.ASSISTANT) {
                const toolBlock = block as ToolUseContentBlock;
                openaiMessages.push({
                  type: 'function_call',
                  call_id: toolBlock.id,
                  name: toolBlock.name,
                  arguments: JSON.stringify(toolBlock.input),
                } as OpenAI.Responses.ResponseFunctionToolCall);
              }
              break;

            case MessageContentType.Thinking: {
              const thinkingBlock = block;
              openaiMessages.push({
                type: 'reasoning',
                id: thinkingBlock.signature,
                encrypted_content: thinkingBlock.thinking,
                summary: [],
              } as OpenAI.Responses.ResponseReasoningItem);
              break;
            }
            case MessageContentType.ToolResult: {
              // Handle tool results as function call outputs
              const toolResult = block;
              // Tool results should be added as separate items in the response

              toolResult.content.forEach((content) => {
                if (content.type === MessageContentType.Text) {
                  openaiMessages.push({
                    type: 'function_call_output',
                    call_id: toolResult.tool_use_id,
                    output: content.text,
                  } as OpenAI.Responses.ResponseInputItem.FunctionCallOutput);
                }

                if (content.type === MessageContentType.Image) {
                  openaiMessages.push({
                    type: 'function_call_output',
                    call_id: toolResult.tool_use_id,
                    output: 'screenshot',
                  } as OpenAI.Responses.ResponseInputItem.FunctionCallOutput);
                  openaiMessages.push({
                    role: 'user',
                    type: 'message',
                    content: [
                      {
                        type: 'input_image',
                        detail: 'high',
                        image_url: `data:${content.source.media_type};base64,${content.source.data}`,
                      },
                    ],
                  } as OpenAI.Responses.ResponseInputItem.Message);
                }
              });
              break;
            }

            default:
              // Handle unknown content types as text
              openaiMessages.push({
                role: 'user',
                type: 'message',
                content: [
                  {
                    type: 'input_text',
                    text: JSON.stringify(block),
                  },
                ],
              } as OpenAI.Responses.ResponseInputItem.Message);
          }
        }
      }
    }

    return openaiMessages;
  }

  // Minimal mapping for Chat Completions (fallback path)
  private formatMessagesForChat(messages: Message[], systemPrompt?: string) {
    const out: any[] = [];

    // prepend system instruction if provided
    if (systemPrompt) {
      out.push({ role: 'system', content: systemPrompt });
    }

    for (const m of messages) {
      const blocks = (m.content as MessageContentBlock[]) || [];

      // gather plain text parts
      const texts = blocks
        .filter((b) => b.type === MessageContentType.Text || (b as any).text)
        .map((b: any) => b.text)
        .filter(Boolean);

      if (texts.length) {
        out.push({
          role: m.role === Role.USER ? 'user' : 'assistant',
          content: texts.join('\n'),
        });
      }

      // map tool results to role='tool' so the model can continue function calling
      for (const b of blocks) {
        if (b.type === MessageContentType.ToolResult) {
          const tr = b as unknown as ToolResultContentBlock;
          const textParts = tr.content
            .map((c: any) =>
              c?.type === MessageContentType.Text ? c.text : '',
            )
            .filter(Boolean);
          out.push({
            role: 'tool',
            tool_call_id: tr.tool_use_id,
            content: textParts.join('\n') || 'tool result',
          });
        }
      }
    }
    return out;
  }

  private formatOpenAIResponse(
    response: OpenAI.Responses.ResponseOutputItem[],
  ): MessageContentBlock[] {
    const contentBlocks: MessageContentBlock[] = [];

    for (const item of response) {
      // Check the type of the output item
      switch (item.type) {
        case 'message':
          // Handle ResponseOutputMessage
          const message = item;
          for (const content of message.content) {
            if ('text' in content) {
              // ResponseOutputText
              contentBlocks.push({
                type: MessageContentType.Text,
                text: content.text,
              } as TextContentBlock);
            } else if ('refusal' in content) {
              // ResponseOutputRefusal
              contentBlocks.push({
                type: MessageContentType.Text,
                text: `Refusal: ${content.refusal}`,
              } as TextContentBlock);
            }
          }
          break;

        case 'function_call':
          // Handle ResponseFunctionToolCall
          const toolCall = item;
          contentBlocks.push({
            type: MessageContentType.ToolUse,
            id: toolCall.call_id,
            name: toolCall.name,
            input: JSON.parse(toolCall.arguments),
          } as ToolUseContentBlock);
          break;

        case 'file_search_call':
        case 'web_search_call':
        case 'computer_call':
        case 'reasoning':
          const reasoning = item as OpenAI.Responses.ResponseReasoningItem;
          if (reasoning.encrypted_content) {
            contentBlocks.push({
              type: MessageContentType.Thinking,
              thinking: reasoning.encrypted_content,
              signature: reasoning.id,
            } as ThinkingContentBlock);
          }
          break;
        case 'image_generation_call':
        case 'code_interpreter_call':
        case 'local_shell_call':
        case 'mcp_call':
        case 'mcp_list_tools':
        case 'mcp_approval_request':
          // Handle other tool types as text for now
          this.logger.warn(
            `Unsupported response output item type: ${item.type}`,
          );
          contentBlocks.push({
            type: MessageContentType.Text,
            text: JSON.stringify(item),
          } as TextContentBlock);
          break;

        default:
          // Handle unknown types
          this.logger.warn(
            `Unknown response output item type: ${JSON.stringify(item)}`,
          );
          contentBlocks.push({
            type: MessageContentType.Text,
            text: JSON.stringify(item),
          } as TextContentBlock);
      }
    }

    return contentBlocks;
  }
}
