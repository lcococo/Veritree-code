NATURALIZE_PROMPT = """You are rewriting a structured process description into a natural business narrative, WITHOUT losing any information.

[STRUCTURED DESCRIPTION]
{template}

[ACTIVITIES] (use exactly these labels; every one must appear verbatim in your narrative)
[{activities}]

[RULES]
- Mention EVERY activity from the list, using its exact label text, at least once.
- Keep the structure fully recoverable from your narrative: keep explicit cue words —
  "then / after / followed by" for order, "at the same time / simultaneously / in parallel"
  for concurrency, "either ... or ... / choice" for exclusive choice,
  "repeat / until / loop back" for loops, "optionally / skip" for silent steps.
- NEVER use ordering words ("then", "after", "followed by", "next") between blocks that
  are concurrent. Concurrent blocks must be announced with "at the same time",
  "simultaneously", "in parallel", or "both ... and ... run concurrently".
- Do NOT add, rename, merge, or drop any activity. No new activities.
- Write natural, business-style prose.
- Output ONLY the narrative.
"""

GENERATION_PROMPT = """Generate a process model for the following process as a JSON tree.

A process tree is built from:
- an activity: a plain string with the activity's label.
- a silent step: the string "tau".
- {"op": "seq", "children": [...]}: children execute in order.
- {"op": "xor", "children": [...]}: exactly one child executes.
- {"op": "par", "children": [...]}: all children run concurrently.
- {"op": "loop", "children": [body, redo]}: execute body, then either exit or execute redo and return to body.

Rules:
- Output ONLY the JSON tree in a json code block.
- Use the labels from ACTIVITIES exactly, character for character.
- Each activity label appears AT MOST ONCE in the tree.
- Use a loop when an activity or phase is repeated.
- seq, xor, and par have two or more children.
- loop has exactly two children.

DESCRIPTION: {description}
ACTIVITIES: [{activities}]
"""
