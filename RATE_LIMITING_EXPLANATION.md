# Rate Limiting Issue - Explanation & Solutions

## What Happened?

You saw many **429 (Too Many Requests)** errors because:

1. **Free Model Rate Limits**: The free OpenRouter model (`openai/gpt-oss-20b:free`) has strict rate limits
2. **Multiple Agents**: Each agent (Paper Analysis, Algorithm Interpretation, API Mapping, Code Integration) makes separate LLM calls
3. **Rapid Requests**: Agents were making requests too quickly, hitting rate limits
4. **No Retry Logic**: The system wasn't handling rate limits gracefully

## What I Fixed

### 1. **Retry Logic with Exponential Backoff**
- Added automatic retry for 429 errors
- Exponential backoff: waits 2s, then 4s, then 8s before giving up
- Reduces failed requests

### 2. **Delays Between Agents**
- Added 1-second delays between agent phases
- Prevents rapid-fire requests that trigger rate limits

### 3. **Better Fallback Handling**
- System now gracefully falls back to rule-based/template code generation when LLM fails
- Tracks whether code came from LLM or fallback

## How to Check Code Source

After processing, check the terminal for:
```
[INFO] Code generation source: LLM
```
or
```
[INFO] Code generation source: FALLBACK
[WARNING] Code was generated using fallback (template-based) method due to LLM rate limits or errors
```

## Solutions for Better Performance

### Option 1: Wait Between Requests (Current)
- The system now waits between agent requests
- Works but slower

### Option 2: Use a Paid Model
- Paid models have higher rate limits
- Update `config/default.yaml`:
  ```yaml
  model: "openai/gpt-4o"  # or another paid model
  ```

### Option 3: Reduce LLM Usage
- Disable LLM for some agents, use rule-based:
  ```yaml
  extractor:
    use_llm: false  # Use rule-based extraction
  
  generator:
    use_llm: false  # Use template-based generation
  ```

### Option 4: Process One Paper at a Time
- Don't upload multiple papers simultaneously
- Wait for one to complete before starting another

## Current Status

✅ **Fixed**: Retry logic and delays added
✅ **Fixed**: Better error handling and fallback
✅ **Fixed**: Code source tracking

The system will now:
- Retry failed requests automatically
- Fall back to rule-based generation if LLM fails
- Show you whether code came from LLM or fallback

