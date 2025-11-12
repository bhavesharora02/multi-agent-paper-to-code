# How to Verify Multi-Agent Pipeline is Running

## Quick Check Methods

### 1. Check Configuration Endpoint
Open in browser: `http://localhost:5000/check-config`

This will show:
- ✅ Multi-Agent Pipeline: ENABLED/DISABLED
- ✅ LLM Extractor: ENABLED/DISABLED  
- ✅ LLM Provider and Model

### 2. Check Console Output
When you upload a PDF, you should see in the terminal:
```
[DEBUG] use_multi_agent_pipeline = True
[DEBUG] Config loaded: True
[DEBUG] Using MULTI-AGENT PIPELINE with PlannerAgent
[PROGRESS] 20%: Phase 1: Analyzing paper content with AI...
[PROGRESS] 30%: Found X algorithm(s) in paper
[PROGRESS] 40%: Phase 2: Interpreting algorithms and mathematical notation...
...
```

### 3. Check Browser Console
Open browser DevTools (F12) → Console tab
You should see status polling messages every 500ms showing progress updates.

### 4. Visual Indicators in UI
- Progress bar should move from 0% to 100%
- Progress text should show detailed agent messages:
  - "Phase 1: Analyzing paper content with AI..."
  - "Found X algorithm(s) in paper"
  - "Phase 2: Interpreting algorithms..."
  - "Phase 3: Mapping algorithms to ML framework APIs..."
  - "Phase 4: Generating complete code implementation..."
- Step indicators should highlight:
  - Step 1: Paper Analysis (20-30%)
  - Step 2: Algorithm Interpretation (40-50%)
  - Step 3: Code Generation (60-80%)
  - Step 4: Complete (80-100%)

## If You See "Using LEGACY/TEMPLATE-BASED method"
This means the multi-agent pipeline is NOT running. Check:
1. `config/default.yaml` has `use_multi_agent_pipeline: true`
2. Flask app was restarted after config changes
3. No errors in console preventing pipeline initialization

## Troubleshooting

### No Progress Updates?
1. Check browser console for JavaScript errors
2. Check Flask terminal for `[PROGRESS]` messages
3. Verify status endpoint: `http://localhost:5000/status/<task_id>`

### Still Using Templates?
1. Verify config file: `cat config/default.yaml | grep use_multi_agent`
2. Restart Flask app
3. Check for import errors in Flask terminal

