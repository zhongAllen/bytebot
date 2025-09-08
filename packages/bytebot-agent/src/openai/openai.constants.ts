import { BytebotAgentModel } from 'src/agent/agent.types';

const isOpenRouter =
  (process.env.OPENAI_API_BASE && process.env.OPENAI_API_BASE.includes('openrouter.ai')) ||
  (process.env.OPENAI_BASE_URL && process.env.OPENAI_BASE_URL.includes('openrouter.ai'));

export const OPENAI_MODELS: BytebotAgentModel[] = isOpenRouter
  ? [
      {
        provider: 'openai',
        name: 'google/gemini-2.5-pro',
        title: 'Gemini 2.5 Pro (OpenRouter)',
        contextWindow: 200000,
      },
      {
        provider: 'openai',
        name: 'anthropic/claude-3.7-sonnet',
        title: 'Claude 3.7 Sonnet (OpenRouter)',
        contextWindow: 200000,
      },
      {
        provider: 'openai',
        name: 'openai/gpt-4o',
        title: 'GPT-4o (OpenRouter)',
        contextWindow: 200000,
      },
    ]
  : [
      {
        provider: 'openai',
        name: 'o3-2025-04-16',
        title: 'o3',
        contextWindow: 200000,
      },
      {
        provider: 'openai',
        name: 'gpt-4.1-2025-04-14',
        title: 'GPT-4.1',
        contextWindow: 1047576,
      },
    ];

export const DEFAULT_MODEL = OPENAI_MODELS[0];