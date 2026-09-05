# LumiSign LASEE — Implementation Evidence Demo

This demo turns LumiSign from a sign-recognition claim into a measurable, explainable sign-coaching system.

## Evidence build added for the Saudi innovation criterion

The upgraded workflow records the complete technical trace instead of relying on descriptive claims:

1. A Saudi Sign Language name and official source URL are attached to every input sample.
2. MediaPipe converts the captured image into 21 three-dimensional landmarks (63 numeric features).
3. Multiple samples from multiple participant codes form a traceable JSONL training dataset.
4. For one sign, a multi-reference LASEE bank is built only from P01 and tested on the held-out P02 participant. For two or more signs, the Random Forest classifier uses a participant holdout when possible.
5. LASEE explains correction errors by hand shape, finger angles, palm orientation and position.
6. Ghost Hand overlays the verified reference on the learner attempt.
7. The governance tab exports one evidence bundle containing the pipeline, dataset summary, model test and SHA-256 audit status.

### Real validation evidence included

The submitted JSONL dataset contains 57 valid samples for two static Saudi signs from two pseudonymous participants. The participant-held-out Random Forest test used 30 training samples and 27 P02 test samples and achieved 85.2% accuracy. Each record contains the official Saudi source URL, 21 three-dimensional landmarks, 63 normalized features, a participant code, and MediaPipe confidence.

The independent participant test uses all 20 P01 samples as the reference bank and keeps all 18 P02 samples out of that bank. Each P02 attempt is matched with its closest valid P01 reference. At the fixed 85% acceptance threshold, 16/18 P02 attempts passed (88.9%). Mean LASEE similarity was 92.0%, median 94.2%, range 78.1–97.5%, and mean MediaPipe detection confidence 97.6%.

This is positive cross-participant consistency evidence for one static Saudi sign. It is not multi-class recognition accuracy. Add at least a second labeled sign before claiming classifier accuracy.

## What the code proves

- MediaPipe computer vision extracts 21 3D hand landmarks from camera/uploaded images.
- LASEE compares a learner attempt with a verified reference sample.
- It reports separate, real geometric scores for hand shape, finger angles, palm orientation, and position.
- Ghost Hand provides an augmented-reality-style visual correction overlay.
- Each AI decision is stored in a tamper-evident SHA-256 hash chain.

The percentages are calculated from the geometry of the captured hand. They are not random or hard-coded.

## Added responsible-AI and verification evidence

- Every LASEE result exposes the raw geometric errors, component weights, and weighted contribution to the final score.
- The classifier returns all class probabilities, the margin between the top two classes, and a confidence-gated `accepted` or `needs_human_review` status.
- The governance page displays a Dataset Card, Model Card, participant-holdout result, and the current SHA-256 audit-chain state.
- A generated evidence manifest fingerprints every included dataset, model, report, validation file, and audit log with SHA-256.
- The training dataset stores geometric landmarks and normalized features rather than retaining the raw camera image.
- The app exports both machine-readable JSON evidence and a reviewer-friendly HTML technical evidence report.

## Run on Windows

Fastest option: double-click `START_LUMISIGN.bat`. It creates the environment on first use and starts the app.

To generate reviewer-ready evidence and rerun the automated tests, double-click `VERIFY_EVIDENCE.bat` after the first setup.

Manual commands:

Open PowerShell inside this folder and run each command separately:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run this once in the same window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then open the local URL shown by Streamlit (normally `http://localhost:8501`).

## Demonstration sequence for the judging evidence

1. Open **Record reference** and capture one correct sign.
2. Open **Evaluate a sign** and capture the same sign correctly.
3. Take a screenshot of the high score and Ghost Hand overlay.
4. Repeat with an intentionally incorrect finger or palm angle.
5. Take a screenshot showing the lower component score and correction instruction.
6. Open **Governance evidence** and take a screenshot of the valid SHA-256 audit chain.

## Run automated tests

```powershell
python -m pytest -q
```

Expected result: `9 passed`.

## Honest technical scope

This evidence build evaluates static hand poses and classifies the two Saudi signs represented in the included dataset. It does not claim to recognize an unlimited Arabic sign vocabulary or dynamic signs. Temporal classification and a larger multi-user dataset remain future work.
