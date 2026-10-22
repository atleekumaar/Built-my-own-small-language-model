"""
Automated Visual QA and End-to-End Verification via Chrome DevTools Protocol (CDP).
Uses the installed Chrome browser and python's websockets library without requiring
external driver downloads or Playwright binaries.
"""
import asyncio
import json
import os
import sys
import base64
import asyncio
import json
import os
import sys
import base64
import time
import subprocess
import urllib.request
import websockets

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ARTIFACT_DIR = r"C:\Users\atuls\.gemini\antigravity-ide\brain\d90e67f5-9b83-40a7-b900-be1f16fe01ff"
CDP_HTTP = "http://127.0.0.1:9222"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def launch_chrome():
    user_data = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "chrome_qa_profile")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={user_data}",
        "http://127.0.0.1:8000/"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for CDP HTTP endpoint
    for _ in range(20):
        try:
            req = urllib.request.urlopen(f"{CDP_HTTP}/json", timeout=1.0)
            data = json.loads(req.read().decode("utf-8"))
            if data:
                print(f"[Chrome] Headless browser ready on port 9222 (PID {proc.pid})")
                return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Chrome failed to expose CDP on port 9222")

async def get_page_ws_url():
    req = urllib.request.urlopen(f"{CDP_HTTP}/json")
    targets = json.loads(req.read().decode("utf-8"))
    for t in targets:
        if t.get("type") == "page" and "8000" in t.get("url", ""):
            return t.get("webSocketDebuggerUrl")
    # If not found with 8000, find any page
    for t in targets:
        if t.get("type") == "page":
            return t.get("webSocketDebuggerUrl")
    raise RuntimeError("No page target found on CDP")

class ChromeController:
    def __init__(self, ws):
        self.ws = ws
        self.msg_id = 0
        self.console_errors = []

    async def send(self, method, params=None):
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        
        while True:
            resp = json.loads(await self.ws.recv())
            # Track console errors
            if resp.get("method") == "Runtime.consoleAPICalled":
                p = resp.get("params", {})
                if p.get("type") == "error":
                    self.console_errors.append(p)
            elif resp.get("method") == "Runtime.exceptionThrown":
                self.console_errors.append(resp.get("params", {}))
                
            if resp.get("id") == self.msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error in {method}: {resp['error']}")
                return resp.get("result", {})

    async def eval_js(self, expression, await_promise=False):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise
        })
        return res.get("result", {}).get("value")

    async def screenshot(self, filename):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(res["data"])
        out_path = os.path.join(ARTIFACT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"[Screenshot] Saved: {filename} ({len(data)} bytes)")
        return out_path

    async def set_viewport(self, width, height, is_mobile=False, scale=1.0):
        await self.send("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": scale,
            "mobile": is_mobile,
        })
        await asyncio.sleep(0.3)

async def run_qa():
    print("=== STARTING AUTOMATED VISUAL QA & E2E VERIFICATION ===")
    ws_url = await get_page_ws_url()
    print(f"Connecting to CDP: {ws_url}")
    
    async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
        cdp = ChromeController(ws)
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        
        # 1. Navigate to Lab
        print("\n--- 1. Navigating to http://127.0.0.1:8000/ ---")
        await cdp.send("Page.navigate", {"url": "http://127.0.0.1:8000/"})
        await asyncio.sleep(2.0)
        
        # Check initial title and status
        title = await cdp.eval_js("document.title")
        status_text = await cdp.eval_js("document.getElementById('connectionStatusText').textContent")
        device_text = await cdp.eval_js("document.getElementById('backendDeviceText').textContent")
        print(f"Title: {title}")
        print(f"Connection Status: {status_text}")
        print(f"Device: {device_text}")
        
        # Screenshot 1: Default Desktop Playground
        await cdp.set_viewport(1440, 900, is_mobile=False)
        await cdp.screenshot("qa_01_desktop_playground_initial.png")

        # 2. Test Tab Navigation
        print("\n--- 2. Testing Tab Navigation ---")
        tabs = [
            ("tabTokenizerBtn", "tokenizerTab", "qa_02_desktop_tokenizer.png"),
            ("tabArchitectureBtn", "architectureTab", "qa_03_desktop_architecture.png"),
            ("tabEvaluationBtn", "evaluationTab", "qa_04_desktop_evaluation.png"),
            ("tabPlaygroundBtn", "playgroundTab", "qa_05_desktop_playground_return.png"),
        ]
        
        for btn_id, pane_id, shot_name in tabs:
            await cdp.eval_js(f"document.getElementById('{btn_id}').click()")
            await asyncio.sleep(0.6)
            is_active = await cdp.eval_js(f"document.getElementById('{pane_id}').classList.contains('active')")
            print(f"Tab {btn_id} -> {pane_id} active: {is_active}")
            await cdp.screenshot(shot_name)

        # 3. Test Generation on Preset 'Abeni,'
        print("\n--- 3. Testing Real Model Generation: 'Abeni,' ---")
        # Ensure we are on playground
        await cdp.eval_js("document.getElementById('tabPlaygroundBtn').click()")
        await asyncio.sleep(0.3)
        
        # Click Generate
        print("Clicking Generate Tokens button...")
        await cdp.eval_js("document.getElementById('generateBtn').click()")
        
        # Wait for generation to complete (polling button disabled state)
        for _ in range(30):
            await asyncio.sleep(0.5)
            is_disabled = await cdp.eval_js("document.getElementById('generateBtn').disabled")
            if not is_disabled:
                break
                
        # Read generation results
        output_text = await cdp.eval_js("document.getElementById('textContentBox').innerText")
        latency_text = await cdp.eval_js("document.getElementById('latencyPill').innerText")
        throughput_text = await cdp.eval_js("document.getElementById('throughputPill').innerText")
        prob_rows_count = await cdp.eval_js("document.querySelectorAll('.prob-row').length")
        step_indicator = await cdp.eval_js("document.getElementById('stepIndicatorText').innerText")
        
        print(f"Output Text: {output_text.strip()[:100]}...")
        print(f"Latency: {latency_text}")
        print(f"Throughput: {throughput_text}")
        print(f"Top-k Candidates Rendered: {prob_rows_count}")
        print(f"Step Indicator: {step_indicator}")
        
        # Screenshot 6: Playground with Real Generation & Probability Explorer
        await cdp.screenshot("qa_06_desktop_generation_result.png")

        # 4. Test Probability Step Scrubber
        print("\n--- 4. Testing Next-Token Probability Scrubber ---")
        # Click step chip 3 if available
        step_chips_count = await cdp.eval_js("document.querySelectorAll('.step-chip').length")
        print(f"Total Step Chips: {step_chips_count}")
        if step_chips_count >= 3:
            await cdp.eval_js("document.querySelectorAll('.step-chip')[2].click()")
            await asyncio.sleep(0.4)
            new_step_text = await cdp.eval_js("document.getElementById('stepIndicatorText').innerText")
            print(f"Scrubbed to: {new_step_text}")
            await cdp.screenshot("qa_07_desktop_scrubber_step3.png")

        # 5. Test Random Sampling Mode with Temperature & Seed
        print("\n--- 5. Testing Sampling Mode Controls ---")
        await cdp.eval_js("""
            const sel = document.getElementById('samplingModeSelect');
            sel.value = 'random';
            sel.dispatchEvent(new Event('change'));
            document.getElementById('temperatureInput').value = '0.70';
            document.getElementById('temperatureInput').dispatchEvent(new Event('input'));
            document.getElementById('seedInput').value = '42';
        """)
        await asyncio.sleep(0.4)
        temp_visible = await cdp.eval_js("document.getElementById('tempControlRow').style.display !== 'none'")
        seed_visible = await cdp.eval_js("document.getElementById('seedControlRow').style.display !== 'none'")
        print(f"Temperature control visible: {temp_visible}, Seed control visible: {seed_visible}")
        await cdp.screenshot("qa_08_desktop_random_sampling_controls.png")

        # Generate with Random Sampling
        await cdp.eval_js("document.getElementById('generateBtn').click()")
        for _ in range(30):
            await asyncio.sleep(0.5)
            is_disabled = await cdp.eval_js("document.getElementById('generateBtn').disabled")
            if not is_disabled:
                break
        await cdp.screenshot("qa_09_desktop_random_sampling_output.png")

        # 6. Test Tokenizer Inspector with Known & Unknown Words
        print("\n--- 6. Testing Tokenizer Inspector with Custom Text ---")
        await cdp.eval_js("document.getElementById('tabTokenizerBtn').click()")
        await asyncio.sleep(0.4)
        
        # Input sentence with known and unknown tokens
        test_sentence = "Abeni visited QuantumCyberX in Lagos."
        await cdp.eval_js(f"""
            document.getElementById('tokenizerTextInput').value = '{test_sentence}';
            document.getElementById('inspectTokenizeBtn').click();
        """)
        await asyncio.sleep(0.6)
        
        tokens_count = await cdp.eval_js("document.getElementById('tokenCountMetric').innerText")
        unk_count = await cdp.eval_js("document.getElementById('unkCountMetric').innerText")
        chip_items = await cdp.eval_js("""
            Array.from(document.querySelectorAll('.token-chip-item')).map(c => ({
                text: c.querySelector('.token-chip-word')?.textContent,
                id: c.querySelector('.token-chip-id')?.textContent,
                isUnk: c.classList.contains('is-unk')
            }))
        """)
        print(f"Token Count: {tokens_count}, UNK Count: {unk_count}")
        print(f"Chips rendered: {chip_items}")
        await cdp.screenshot("qa_10_desktop_tokenizer_analysis.png")

        # 7. Mobile Viewport Testing (375x812 - iPhone X/12/13/14)
        print("\n--- 7. Testing Mobile Viewport (375x812) ---")
        await cdp.set_viewport(375, 812, is_mobile=True, scale=2.0)
        
        # Mobile Playground
        await cdp.eval_js("document.getElementById('tabPlaygroundBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.screenshot("qa_11_mobile_playground.png")
        
        # Mobile Tokenizer
        await cdp.eval_js("document.getElementById('tabTokenizerBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.screenshot("qa_12_mobile_tokenizer.png")
        
        # Mobile Architecture
        await cdp.eval_js("document.getElementById('tabArchitectureBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.screenshot("qa_13_mobile_architecture.png")
        
        # Mobile Evaluation
        await cdp.eval_js("document.getElementById('tabEvaluationBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.screenshot("qa_14_mobile_evaluation.png")

        # 8. Reset to Desktop and Check Phase 4 Multi-Model Switching & BPE Capabilities
        await cdp.set_viewport(1440, 900, is_mobile=False)
        print("\n--- 8. Phase 4 Multi-Model Switching to Byte-Level BPE ---")
        
        # Switch model via dropdown
        await cdp.eval_js("""
            const sel = document.getElementById('modelSelect');
            sel.value = 'slm_bpe_val';
            sel.dispatchEvent(new Event('change'));
        """)
        await asyncio.sleep(1.0)
        
        badge_text = await cdp.eval_js("document.getElementById('brandBadge').innerText")
        param_tag_text = await cdp.eval_js("document.getElementById('headerParamTag').innerText")
        print(f"[Model Switch] Badge: '{badge_text}', Header Tag: '{param_tag_text}'")
        
        # Verify Tokenizer Tab with BPE
        print("\n--- 9. Phase 4 BPE Tokenizer Zero-UNK Verification ---")
        await cdp.eval_js("document.getElementById('tabTokenizerBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.eval_js("""
            const tInput = document.getElementById('tokenizerInput');
            tInput.value = 'SuperQuantum12345';
            document.getElementById('tokenizeBtn').click();
        """)
        await asyncio.sleep(0.8)
        bpe_unk = await cdp.eval_js("document.getElementById('unkCountMetric').innerText")
        bpe_chips = await cdp.eval_js("""
            Array.from(document.querySelectorAll('.token-chip-item')).map(c => ({
                text: c.querySelector('.token-chip-word')?.textContent,
                id: c.querySelector('.token-chip-id')?.textContent,
                isUnk: c.classList.contains('is-unk')
            }))
        """)
        print(f"[BPE Tokenizer] UNK Count for 'SuperQuantum12345': {bpe_unk}")
        print(f"[BPE Tokenizer] Chips: {bpe_chips}")
        await cdp.screenshot("qa_15_bpe_tokenizer_zero_unk.png")
        
        # Generate with BPE
        print("\n--- 10. Phase 4 BPE Playground Generation ---")
        await cdp.eval_js("document.getElementById('tabPlaygroundBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.eval_js("""
            document.getElementById('promptInput').value = 'Kilimanjaro is';
            document.getElementById('generateBtn').click();
        """)
        await asyncio.sleep(2.0)
        bpe_gen_text = await cdp.eval_js("document.getElementById('outputText').innerText")
        print(f"[BPE Generation] Output text: {bpe_gen_text}")
        await cdp.screenshot("qa_16_bpe_generation.png")
        
        # Check Evaluation Tab Comparison Table
        print("\n--- 11. Phase 4 Experiment Comparison Table ---")
        await cdp.eval_js("document.getElementById('tabEvaluationBtn').click()")
        await asyncio.sleep(0.5)
        await cdp.screenshot("qa_17_experiment_comparison_table.png")

        # 12. Check Console Errors
        print("\n--- 12. Console Error Audit ---")
        print(f"Total Console Errors Caught: {len(cdp.console_errors)}")
        for err in cdp.console_errors:
            print(f"  Console Error: {err}")

    print("\n=== VISUAL QA & E2E VERIFICATION COMPLETED ===")

if __name__ == "__main__":
    chrome_proc = launch_chrome()
    try:
        asyncio.run(run_qa())
    finally:
        print("[Chrome] Terminating headless Chrome process...")
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except Exception:
            chrome_proc.kill()

