🧠 Multi-Agent Translation Pipeline Experiment
(Designed for Claude Code — Using the Claude Code Agent Schema)

This document describes the assignment requirements and execution plan for building a multi-agent translation system using Claude Code agents. The system will run a full experiment pipeline that translates text through multiple languages, measures semantic drift using embeddings, and visualizes the effect of spelling errors on translation stability.

📦 Task Overview (Schema of the Entire Workflow)

The project consists of the following main stages:

Build Proper Claude Code Agents

Define 3 translation agents using the Claude Code Agent Schema.

Each agent has its own skill: EN→FR, FR→HE, HE→EN.

Agents must be fully defined with:

name

description

skills

input_schema / output_schema

Any relevant transformation rules

Prepare Valid Input Sentences

At least 15 words each.

Include ≥25% spelling errors.

Prepare multiple versions of the sentence with increasing error percentages (0–50%).

Run the End-to-End Experiment

A controller program (or a 4th agent) will:

Send the sentence to Agent 1

Pipe the output into Agent 2

Pipe the output into Agent 3

Collect all intermediate translations

Store the final English output

Record all data for analysis

This controller is responsible for the entire orchestration and will be executed in Claude Code as well.

Measure Embedding Distance in Python

Compute embeddings for:

Original English sentence

Final translated English sentence

Measure vector distance (cosine or Euclidean).

Repeat for all spelling-error levels.

Generate Graphs

Plot error percentage (0–50%) vs. embedding distance.

Graphs will be produced in Python.

Graphs must be included in the submission.

🧩 Architecture and Components
⚙️ Translation Agents (Claude Code Agent Schema)

You must define three agents, each following the Claude Code agent JSON/YAML schema:

1. Agent: English → French

Description: Translates English sentences into French.

Input schema: { "text": "string" }

Output schema: { "translated_text": "string" }

2. Agent: French → Hebrew

Description: Translates French sentences into Hebrew.

Input schema: { "text": "string" }

Output schema: { "translated_text": "string" }

3. Agent: Hebrew → English

Description: Translates Hebrew sentences back into English.

Input schema: { "text": "string" }

Output schema: { "translated_text": "string" }

Each agent must be explicitly declared as a Claude Code agent.

🧠 Experiment Controller (Program or 4th Agent)

A central execution program must be created. It may be:

A Python script executed inside Claude Code
or

A dedicated Claude Code Agent (recommended)

This controller must:

Accept the prepared English sentence

Pass it through all 3 translation agents sequentially

Store:

Original sentence

Intermediate translations

Final English sentence

Error level

Save all outputs into a structured dictionary or CSV

Export the final dataset to Python for embedding computation

This ensures the experiment is fully automated and Claude Code can run it end-to-end.

✍️ Input Sentence Specifications

Each experiment requires:

≥15 words

≥25% spelling errors

Multiple versions of the sentence with:

0% errors

10% errors

20% errors

25% errors

30% errors

40% errors

50% errors

You must include these sentences in the final submission.

🧮 Embedding Distance Measurement

After the experiment runs:

Import results into Python (inside Claude Code)

Compute embeddings for original & final sentences

Compute vector distance

Store distances in a table

Example metric:

Cosine distance

Euclidean distance

📊 Graph Requirements

Create graphs showing:

X-axis: % spelling errors

Y-axis: embedding distance

The graph must clearly illustrate how translation robustness degrades as spelling errors increase.

Graphs must be included in the submitted materials.

📁 Submission Requirements

You must submit:

1. Input Sentences

All versions (0%–50% errors)

Word counts

2. Agent Definitions

Claude Code agent schemas for all 3 translation agents

Agent schema for the experiment controller

3. Pipeline Outputs

Intermediate translations

Final English outputs

Recorded spelling error levels

4. Embedding Distances

Table of distances per error level

5. Graphs

Error % vs. vector distance

6. (Optional but recommended)

Python scripts used for embeddings & graph creation

📌 Note

This entire specification is intended to be given to Claude Code directly so that the system can execute the experiment automatically.