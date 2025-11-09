# ElevenLabs MCP Enhanced

<div class="title-block" style="text-align: center;" align="center">

  [![npm version](https://img.shields.io/npm/v/elevenlabs-mcp-enhanced.svg?style=for-the-badge&logo=npm&labelColor=000)](https://www.npmjs.com/package/elevenlabs-mcp-enhanced)
  [![npm downloads](https://img.shields.io/npm/dm/elevenlabs-mcp-enhanced.svg?style=for-the-badge&logo=npm&labelColor=000)](https://www.npmjs.com/package/elevenlabs-mcp-enhanced)
  [![Discord Community](https://img.shields.io/badge/discord-@elevenlabs-000000.svg?style=for-the-badge&logo=discord&labelColor=000)](https://discord.gg/elevenlabs)
  [![License](https://img.shields.io/badge/license-MIT-000000.svg?style=for-the-badge&logo=opensource&labelColor=000)](LICENSE)

</div>

<p align="center">
  <strong>Enhanced fork of the official ElevenLabs MCP server</strong> with additional conversational AI features including conversation history and transcript retrieval.
</p>

<p align="center">
  This enhanced version is developed and maintained by <strong>Boris Djordjevic</strong> and the <strong>199 Longevity</strong> team.
</p>

## 📑 Table of Contents

- [🚀 What's New](#-whats-new-in-this-fork)
- [🚀 Quick Install](#-quick-install)
- [📋 Requirements](#-requirements)
- [⚙️ Setup Guide](#quickstart-with-claude-desktop)
- [💡 Example Usage](#example-usage)
- [🛠️ Development](#contributing)
- [👥 Credits](#credits)

## 🚀 What's New in This Fork

This enhanced version adds critical conversational AI features missing from the original:

### 🤖 AI-Friendly Improvements (v1.0.0)
- **✅ Official v3 API**: Now uses official ElevenLabs endpoints - no proxy needed!
- **🎯 Smart Voice Defaults**: `search_voices()` now returns common working voices instantly
- **📚 Educational Error Messages**: Errors guide AI agents to success with examples
- **💡 Clear Tool Guidance**: No more confusion about single vs multi-speaker tools
- **🎤 Accurate v3 Voice IDs**: All 20 v3-optimized voices now have correct IDs and descriptions
- **🏯 Auto-Split Long Dialogues**: Automatically splits dialogues over 3000 chars into multiple files
- **🎯 Auto-Adjust Stability**: Invalid stability values auto-round to nearest valid option (0.0, 0.5, 1.0)
- **🏷️ Smart Tag Simplification**: Complex tags auto-convert to valid v3 tags for better quality
- **⏱️ Dynamic Timeouts**: Prevents timeouts on complex dialogues by calculating appropriate wait times

### 🆕 ElevenLabs v3 Model Support (Official)
- **🎭 Enhanced Expressiveness**: Use the official v3 model with `model="v3"` parameter
- **🎤 Audio Tags**: Add emotions and sound effects like `[thoughtful]`, `[crying]`, `[laughing]`, `[piano]`
- **👥 Multi-Speaker Dialogue**: Generate natural conversations between multiple speakers
- **✨ Dialogue Enhancement**: Automatically enhance your dialogue with proper formatting and tags
- **🌍 70+ Languages**: v3 supports multilingual synthesis with emotional control
- **✅ Official API**: Now uses the official ElevenLabs text-to-dialogue endpoint

### 🎙️ Conversational AI Features
- **Conversation History**: Retrieve full conversation details including transcripts
- **📝 Transcript Access**: Get conversation transcripts in multiple formats (plain, timestamps, JSON)
- **⏳ Real-time Monitoring**: Wait for ongoing conversations to complete and retrieve results
- **🔍 Conversation Search**: List and filter conversations by agent, status, and more
- **🎨 Improved Formatting**: Consistent formatting across all list operations

## About

This is an enhanced fork of the official ElevenLabs <a href="https://github.com/modelcontextprotocol">Model Context Protocol (MCP)</a> server that enables interaction with powerful Text to Speech and audio processing APIs. This server allows MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to generate speech, clone voices, transcribe audio, manage conversational AI agents, and now retrieve conversation history.

## 🚀 Quick Install

### Zero Install (Recommended)

**No installation required!** Just use npx:

```bash
npx elevenlabs-mcp-enhanced --api-key YOUR_API_KEY
```

### Global Install

Install once, use everywhere:

```bash
npm install -g elevenlabs-mcp-enhanced
elevenlabs-mcp-enhanced --api-key YOUR_API_KEY
```

### Environment Variable

Set your API key once:

```bash
export ELEVENLABS_API_KEY="your-api-key"
npx elevenlabs-mcp-enhanced
```

## 📋 Requirements

- **Node.js 16+** (for npm/npx)
- **Python 3.11+** (automatically managed by the npm package)
- **ElevenLabs API Key** - Get one at [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys)

## Quickstart with Claude Desktop

### Option 1: Using npm/npx (Recommended - No installation required!)

1. Get your API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys). There is a free tier with 10k credits per month.
2. Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```json
{
  "mcpServers": {
    "ElevenLabs": {
      "command": "npx",
      "args": ["elevenlabs-mcp-enhanced"],
      "env": {
        "ELEVENLABS_API_KEY": "<insert-your-api-key-here>"
      }
    }
  }
}
```

That's it! No installation needed - npx will automatically download and run the server.

### Option 2: Using Python (Original method)

If you prefer the original Python installation:

1. Get your API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys).
2. Install from GitHub:
   ```bash
   pip install git+https://github.com/199-biotechnologies/elevenlabs-mcp-enhanced.git
   ```
3. Configure Claude Desktop with:
   ```json
   {
     "mcpServers": {
       "ElevenLabs": {
         "command": "python",
         "args": ["-m", "elevenlabs_mcp"],
         "env": {
           "ELEVENLABS_API_KEY": "<insert-your-api-key-here>"
         }
       }
     }
   }
   ```

If you're using Windows, you will have to enable "Developer Mode" in Claude Desktop to use the MCP server. Click "Help" in the hamburger menu at the top left and select "Enable Developer Mode".

## Other MCP clients

### Using npm/npx:
For other clients like Cursor and Windsurf, you can run the server directly:
```bash
npx elevenlabs-mcp-enhanced --api-key YOUR_API_KEY
```

### Using Python:
1. `pip install elevenlabs-mcp`
2. `python -m elevenlabs_mcp --api-key={{PUT_YOUR_API_KEY_HERE}} --print` to get the configuration. Paste it into appropriate configuration directory specified by your MCP client.

That's it. Your MCP client can now interact with ElevenLabs through these tools:

## Example usage

⚠️ Warning: ElevenLabs credits are needed to use these tools.

Try asking Claude:

- "Create an AI agent that speaks like a film noir detective and can answer questions about classic movies"
- "Generate three voice variations for a wise, ancient dragon character, then I will choose my favorite voice to add to my voice library"
- "Convert this recording of my voice to sound like a medieval knight"
- "Create a soundscape of a thunderstorm in a dense jungle with animals reacting to the weather"
- "Turn this speech into text, identify different speakers, then convert it back using unique voices for each person"

### 🆕 v3 Model - Quick Start Guide

**🎯 DECISION TREE:**
1. **Single speaker?** → Use `text_to_speech` with `model="v3"`
2. **Multiple speakers?** → Use `text_to_dialogue` (automatically v3)
3. **Need tag examples?** → Call `fetch_v3_tags()` first

**📋 RECOMMENDED WORKFLOW FOR AI:**
```
1. User: "Create an emotional story with sound effects"
2. AI: fetch_v3_tags() → Gets list of available tags
3. AI: search_voices("v3") → Gets v3-optimized voices
4. AI: text_to_dialogue(...) → Creates the story
```

**Single Speaker Examples (text_to_speech):**
- "Generate: '[thoughtful] The universe is vast... [piano] ...and full of mysteries.'"
- "Create narration with: '[whispering] Secret message [footsteps] [door creaking]'"

**Multi-Speaker Examples (text_to_dialogue - ALWAYS v3):**
```python
# Simple conversation
inputs = [
    {"text": "How are you?", "voice_name": "James"},
    {"text": "I'm great!", "voice_name": "Jane"}
]

# With emotion tags
inputs = [
    {"text": "[excited] I found treasure!", "voice_name": "James"},
    {"text": "[skeptical] Really? [pause] Where?", "voice_name": "Jane"}
]
```

**⚠️ v3 Requirements:**
- Stability: MUST be 0.0, 0.5, or 1.0 (no other values!)
- Best voices: James, Jane, Sarah, Mark, etc. (search "v3" to find them)
- Always check fetch_v3_tags() for available audio tags

### 🆕 New Conversation Features

With the enhanced conversation tools, you can now:

- "Get the conversation transcript from conversation ID abc123" (automatically waits for completion)
- "List all conversations from my agent and show me the completed ones"
- "Get conversation xyz789 immediately without waiting" (set wait_for_completion=false)
- "Show me all conversations in JSON format with timestamps"
- "Get the conversation history including analysis data"

**Note:** The `get_conversation` tool now waits for conversations to complete by default (up to 5 minutes), ensuring you always get the full transcript.

## Optional features

You can add the `ELEVENLABS_MCP_BASE_PATH` environment variable to the `claude_desktop_config.json` to specify the base path MCP server should look for and output files specified with relative paths.

### ✅ v3 Model - Now Officially Available!

The v3 model is now officially available through the ElevenLabs API! No proxy or special access needed - just use your regular API key.

**What's New:**
- Official `eleven_v3` model ID
- Text-to-dialogue endpoint at `/v1/text-to-dialogue`
- 70+ language support
- 3,000 character limit per request
- Enhanced emotional expressiveness

**Usage:**
Simply set `model="v3"` in `text_to_speech()` or use `text_to_dialogue()` for multi-speaker content. The server now uses the official API endpoints.

## Contributing

If you want to contribute or run from source:

1. Clone the repository:

```bash
git clone https://github.com/elevenlabs/elevenlabs-mcp
cd elevenlabs-mcp
```

2. Create a virtual environment and install dependencies [using uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and add your ElevenLabs API key:

```bash
cp .env.example .env
# Edit .env and add your API key
```

4. Run the tests to make sure everything is working:

```bash
./scripts/test.sh
# Or with options
./scripts/test.sh --verbose --fail-fast
```

5. Install the server in Claude Desktop: `mcp install elevenlabs_mcp/server.py`

6. Debug and test locally with MCP Inspector: `mcp dev elevenlabs_mcp/server.py`

## Troubleshooting

Logs when running with Claude Desktop can be found at:

- **Windows**: `%APPDATA%\Claude\logs\mcp-server-elevenlabs.log`
- **macOS**: `~/Library/Logs/Claude/mcp-server-elevenlabs.log`

### Timeouts when using certain tools

Certain ElevenLabs API operations, like voice design and audio isolation, can take a long time to resolve. When using the MCP inspector in dev mode, you might get timeout errors despite the tool completing its intended task.

This shouldn't occur when using a client like Claude.

### MCP ElevenLabs: spawn uvx ENOENT

If you encounter the error "MCP ElevenLabs: spawn uvx ENOENT", confirm its absolute path by running this command in your terminal:

```bash
which uvx
```

Once you obtain the absolute path (e.g., `/usr/local/bin/uvx`), update your configuration to use that path (e.g., `"command": "/usr/local/bin/uvx"`). This ensures that the correct executable is referenced.

## Credits

### Enhanced Fork
- **Boris Djordjevic** - Lead Developer
- **199 Longevity Team** - Development and Testing

### Original ElevenLabs MCP Server
- **Jacek Duszenko** - jacek@elevenlabs.io
- **Paul Asjes** - paul.asjes@elevenlabs.io
- **Louis Jordan** - louis@elevenlabs.io
- **Luke Harries** - luke@elevenlabs.io

This enhanced fork builds upon the excellent foundation created by the ElevenLabs team, adding critical conversational AI features for improved agent interaction and monitoring.

## License

This project maintains the same MIT license as the original ElevenLabs MCP server. See [LICENSE](LICENSE) for details.
