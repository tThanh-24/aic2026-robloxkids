# Video Retrieval Task Catalog

Standardized definitions of the four benchmark tasks: **KIS**, **AVS**, **VQA**, **KISC**.
Use these terms consistently across docs, slides, and code: **Keyframe**, **Ground Truth**, **Ranked List**, **Temporal Reasoning**.

## At a Glance

| Task | Type | Goal | Input | Output | Scoring |
|------|------|------|-------|--------|---------|
| **KIS** — Known-Item Search | Single-target | Pinpoint one exact moment | Text description or example clip | One video segment / Keyframe | Top-1 / Top-5 accuracy |
| **AVS** — Ad-hoc Video Search | Set-retrieval | Collect all matching moments | Short semantic description | Ranked List of segments | Top-N scoring across cutoffs |
| **VQA** — Video Question Answering | Reasoning | Answer a question about a video | Long video + question | Short text answer | Answer accuracy |
| **KISC** — Conversational KIS | Interactive | Converge on the target via dialogue | Multi-turn conversation | Refined result after each turn | Accuracy at end of dialogue |

**Single-target vs. set-retrieval** — the key distinction:

- **KIS** has exactly **one Ground Truth**. Only precision at the very top of the Ranked List matters: rank 1 (or within Top-5) or fail.
- **AVS** has **many relevant segments**. Recall across the whole Ranked List matters: every relevant segment retrieved high in the Top-N scores points.

---

## KIS — Known-Item Search

> **Core:** locate the single correct answer (**Ground Truth**) in the database.

- User describes one specific, previously-seen fragment from memory.
- System must return the exact video segment or **Keyframe** containing that fragment.
- Success demands absolute precision at the top of the Ranked List (**Top-1 / Top-5 accuracy**).
- Hard cases: tiny objects, 1–2 second fleeting actions, contextual clues — keyword search alone floods the result with wrong hits.

**Two variants:**

| Variant | Given | Task |
|---------|-------|------|
| **Video KIS** | A clip sampled from the dataset | Identify which video / Keyframe the clip comes from (near-duplicate matching) |
| **Textual KIS** | No clip — only a text description | Bridge the language-to-vision gap to pinpoint the same unique target |

**Example query:** *"the scene where someone drops a teddy-bear keychain with a pink lock while passing a fruit stall."*

---

## AVS — Ad-hoc Video Search

> **Core:** broad semantic retrieval — collect **every** scene that satisfies a topic.

- Query is generic: it names a *class* of moments, never one specific segment.
- System scans the full database and returns a **Ranked List** of segments ordered by descending similarity.
- **Top-N scoring** rewards relevant segments placed high in the Ranked List.
- Requires understanding interactions and relations ("adult *instructs* child to *water* flowers"), not just isolated object matches.
- Must respect negative constraints — e.g. "grilling a burger at home" does not match *"eating fast food alone in a restaurant."*

**Example query:** *"find all moments of someone eating fast food alone in a restaurant."*

---

## VQA — Video Question Answering

> **Core:** understand the video and answer like a human — the stepping stone to a true virtual assistant.

- Input: one long video **plus** a single natural-language question.
- Output: a short, precise **text answer** — not an image.
- Skills under test: attribute binding, **Temporal Reasoning**, and **counting**.
- Typical reasoning chain: identify the subject → locate the event → count → track the subsequent action chain.

**Example:** given a 2-minute birthday-party video — *"How many candles does the boy in the red superhero shirt blow out, and who gives him a fruit afterwards?"*

---

## KISC — Conversational KIS

> **Core:** a multi-turn clarification loop — the assistant **asks back** instead of guessing from one vague command.

- Opening queries are under-specified ("video where I met an old friend last week") and match hundreds of candidates.
- Assistant asks targeted clarification questions to narrow the search scope.
- Each user reply contributes **metadata / trait filters**: time window, location, gender, clothing color.
- System combines filters → re-ranks → returns the most accurate result; the loop repeats until the target is found.

**Example loop:**

| Turn | Speaker | Content | Effect |
|------|---------|---------|--------|
| 1 | User | "Find the video where I met an old friend last week." | Too vague — hundreds of candidates |
| 2 | Assistant | "Was the meeting indoors or outdoors? Was the friend male or female?" | Narrows scope |
| 3 | User | "At an outdoor café; he wore a blue shirt." | Adds location + trait filters |
| 4 | Assistant | Filters: time (last week) + place (outdoor café) + traits (male, blue shirt) → returns top result | Converges |

---

## Standardized Terminology

| Term | Definition |
|------|------------|
| **Keyframe** | Representative still frame extracted from a shot; the atomic retrieval unit for KIS/AVS |
| **Ground Truth (GT)** | The annotated correct answer for a query — the unique target (KIS) or the relevant segment set (AVS) |
| **Ranked List** | System output: candidate segments ordered by descending relevance score |
| **Temporal Reasoning** | Chaining visual evidence across time within one video — order, duration, tracking, and counting |
