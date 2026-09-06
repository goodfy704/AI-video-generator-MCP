# AI Video Generator

An MCP-based AI video generation server built with **Python** and **FastMCP**.

The project is being developed incrementally, starting with a local proof-of-concept and eventually evolving into a remotely accessible, containerized AI video-generation service.

The initial goal is to evaluate local LLMs such as **Qwen3.5 27B** and **gpt-oss-20B** as agents capable of controlling an AI video-generation workflow through MCP tools.

---

## Project Goals

The project aims to provide an MCP interface for AI video generation, allowing an LLM to perform actions such as:

* Create a video from a prompt
* Check video-generation status
* Retrieve generated videos
* Cancel video-generation jobs
* Eventually work with different video-generation backends

The MCP interface should remain independent from the underlying video-generation implementation.

This allows the project to evolve from a local GPU-based prototype into a remotely deployed service without redesigning the MCP tools.

---

# Project setup

You should have installed python 3.14.7. You can download it from Python Install Manager here: https://www.python.org/downloads/

Install LM studio: https://lmstudio.ai/

Clone repository to your local machine

Open cloned repository folder with terminal and create local virtual environment:

```
py -m venv .venv
```

Go to mcp.json and change cwd to your folder where you have the cloned repository

Run from root folder:

```
uv sync
```

Go to LM studio settings -> Connected Apps -> Custom MCP. Fill the as shown in the screenshot except use your cloned repository folder in Working directory.

https://imgur.com/a/ieP7waX

From vsc run:

```
uv run python -m video_mcp.server
```

The server must be started as a module (`-m`) from the project root, not as a
loose script path, so that `video_mcp` imports resolve.

In LM studio you should see connected status with tools available

Go to LM studio settings -> Library. Change model install folder to your bigger ssd

https://imgur.com/a/uZWjYU0

Go to LM studio settings -> Explore -> search for qwen 3.5 27B GGUF -> download unsloth version

https://imgur.com/a/5F58WzX

TEST if LLM is working

In chat write:

```
Create a 10 second video of a futuristic city at night in 16:9.
```

It should call the create_video() tool.

The mock job does not finish instantly. It stays queued for ~2 seconds, runs for
~8 seconds, and only then reports `completed`. A model that handles the workflow
correctly will poll `get_video_status` and then call `get_video_result`.

The timings can be adjusted for an evaluation run:

```
AI_VIDEO_MOCK_QUEUED_SECONDS=0
AI_VIDEO_MOCK_RUNNING_SECONDS=30
```

## Running the tests

```
uv run pytest
uv run ruff check .
```


# Architecture

The project will be developed in several stages.

### Current target architecture

```text
                LM Studio
                    │
             Local LLM model
          ┌─────────┴─────────┐
          │                   │
     Qwen3.5 27B        gpt-oss-20B
          │                   │
          └─────────┬─────────┘
                    │
                MCP Client
                    │
                  stdio
                    │
                    ▼
              FastMCP Server
                    │
                    ▼
             Video Service
                    │
                    ▼
            Local GPU / Backend
```

The LLM is responsible for understanding the user's request and deciding which MCP tools to use.

The actual video generation is performed by a separate video-generation backend.

---

# Roadmap

## Phase 1 — Local MVP / Proof of Concept

**Status: Planned**

The first phase focuses entirely on proving that the concept works.

There will be:

* No Docker
* No HTTPS
* No remote deployment
* No Terraform
* No Ansible
* No Jenkins
* No VM

The MCP server will run locally using **stdio transport**.

The video-generation workload will initially use the GPU and other resources available on the developer's PC.

### Initial architecture

```text
LM Studio
    │
    │ Local LLM
    ▼
MCP Client
    │
    │ stdio
    ▼
FastMCP Server
    │
    ▼
Video-generation tools
    │
    ▼
Developer PC GPU
```

### Initial MCP tools

The first version will provide a minimal set of tools, for example:

```text
create_video()
get_video_status()
get_video_result()
cancel_video()
```

The first implementation may use a mock/fake video backend.

This is intentional.

Before connecting an actual video-generation model, the project should establish that the selected LLM can reliably:

1. Understand the available MCP tools
2. Select the correct tool
3. Generate valid tool arguments
4. Handle returned job IDs
5. Check job status
6. Handle errors
7. Complete a multi-step video-generation workflow

### LLM evaluation

The initial models to evaluate are:

* Qwen3.5 27B
* gpt-oss-20B

Additional models may be tested later.

The models will be tested through **LM Studio** using the same MCP server and the same tool definitions.

The goal is to determine which model provides the best combination of:

* Tool-calling reliability
* Reasoning
* Parameter accuracy
* Context handling
* Speed
* Resource consumption
* Error recovery
* Overall reliability as an MCP agent

---

# Phase 2 — Remote MVP

**Status: Planned**

Once the local proof-of-concept works, the MCP server will be adapted for remote access.

The transport will move from:

```text
stdio
```

to:

```text
Streamable HTTP
```

The service will eventually be exposed through:

```text
HTTPS
```

### Target architecture

```text
                Internet
                    │
                   HTTPS
                    │
                    ▼
             Streamable HTTP
                    │
                    ▼
              FastMCP Server
                    │
                    ▼
             Video Backend
                    │
                    ▼
              GPU / Compute
```

This phase introduces concerns that are not necessary during local development, including:

* HTTPS/TLS
* Authentication
* Authorization
* Secrets management
* Network security
* Request validation
* Logging
* Error handling
* Rate limiting
* Remote configuration

The goal is to make the MCP server usable remotely while keeping the underlying MCP tool interface stable.

---

# Phase 3 — Production MVP

**Status: Planned**

After the remote MVP has been validated, the application will be moved away from the developer's personal PC and deployed to dedicated infrastructure.

This phase introduces:

* Docker
* Virtual machines
* Dedicated GPU compute
* Persistent storage
* Environment configuration
* Service management

### Target architecture

```text
                    Internet
                       │
                      HTTPS
                       │
                       ▼
                 ┌───────────┐
                 │    VM     │
                 │           │
                 │ FastMCP   │
                 │ Server    │
                 └─────┬─────┘
                       │
                       ▼
                 Video Backend
                       │
                       ▼
                  GPU Compute
```

Docker will package the application and its Python dependencies into a reproducible environment.

The development environment and production environment should remain consistent as much as practical.

---

# Phase 4 — Infrastructure & Automation

**Status: Planned**

Once the application is running reliably on dedicated infrastructure, infrastructure automation and CI/CD will be introduced.

Potential technologies include:

* **Terraform** — infrastructure provisioning
* **Ansible** — server configuration and deployment
* **Jenkins** — CI/CD and deployment automation
* GitHub — source control and collaboration

### Target workflow

```text
Developer
    │
    ▼
GitHub
    │
    ▼
CI / Tests
    │
    ▼
Jenkins
    │
    ├── Terraform
    │       │
    │       ▼
    │     Infrastructure
    │
    └── Ansible
            │
            ▼
        VM / Services
            │
            ▼
          Docker
            │
            ▼
       FastMCP Server
            │
            ▼
       Video Backend
            │
            ▼
         GPU Worker
```

The purpose of this phase is to make the system reproducible, deployable, and maintainable rather than manually configured.

---

# Development Strategy

The project intentionally follows an incremental approach.

We do **not** want to solve infrastructure problems before the core application is proven.

The progression is:

```text
1. Prove the MCP concept
        ↓
2. Test local LLMs
        ↓
3. Connect real video generation
        ↓
4. Enable remote access
        ↓
5. Move to dedicated infrastructure
        ↓
6. Containerize
        ↓
7. Automate infrastructure and deployment
```

Each phase should produce a working system before the next layer of complexity is introduced.

---

# Technology Stack

## Initial

* Python
* FastMCP 4.x
* MCP
* LM Studio
* Qwen3.5 27B
* gpt-oss-20B
* Local GPU

## Development

* `uv`
* `pyproject.toml`
* Python virtual environment
* Git
* GitHub
* pytest
* Ruff

## Later

* Streamable HTTP
* HTTPS
* Docker
* Virtual machines
* GPU infrastructure
* Terraform
* Ansible
* Jenkins

The exact video-generation model and backend will be selected after the initial MCP/LLM proof-of-concept.

---

# Project Structure

The project is expected to follow a structure similar to:

```text
AI-video-generator/
│
├── video_mcp/
│   ├── __init__.py
│   ├── server.py        MCP tool layer (thin)
│   ├── video.py         video-generation backend
│   └── models.py        request/response models
│
├── tests/
│   ├── conftest.py
│   ├── test_server.py   tool layer, via an in-memory MCP client
│   └── test_video.py    backend job lifecycle
│
├── mcp.json
├── pyproject.toml
├── uv.lock
├── .gitignore
└── readme.md
```

The package is named `video_mcp`, not `mcp`. A local package called `mcp` shadows
the installed `mcp` SDK that FastMCP depends on, which breaks both the server and
the test suite when anything runs from the project root.

Subpackages (`tools/`, `services/`, `models/`) can be introduced later if the
flat modules grow. Docker-related files are not needed during Phase 1 and will
become relevant during the deployment phase.

---

# Development Environment

A dedicated Python environment will be used during development.

The project should not install its dependencies globally into the developer's system Python installation.

`uv` will be used to manage the project environment and dependencies.

For example:

```bash
uv sync
```

This creates/updates the project's isolated environment based on `pyproject.toml` and `uv.lock`.

The virtual environment should not be committed to Git.

---

# MCP Transport

## Phase 1

```text
stdio
```

stdio is used because the MCP server is running locally and the main objective is rapid development and testing.

## Phase 2+

```text
Streamable HTTP
```

Streamable HTTP will be introduced when the MCP server needs to be accessed remotely.

The MCP tools themselves should remain largely independent of the transport.

For example:

```python
@mcp.tool
def create_video(
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
): ...
```

The same logical tool should be usable regardless of whether the MCP server is accessed through stdio or Streamable HTTP.

---

# Design Principles

### 1. Keep the MCP layer thin

MCP tools should expose a clean interface to the LLM.

Complex video-generation logic should live in services/backend components rather than directly inside the MCP tool implementation.

### 2. Keep the video backend replaceable

The MCP server should not be tightly coupled to one video-generation implementation.

Possible future backends include:

```text
Local video model
ComfyUI
Remote video-generation API
Dedicated GPU worker
```

### 3. Keep the LLM replaceable

The MCP server should not be designed around a specific LLM.

The same MCP tools should be testable with:

```text
Qwen3.5 27B
gpt-oss-20B
Other local models
Future models
```

### 4. Introduce infrastructure only when needed

The project will start locally and become progressively more production-oriented.

There is no need to introduce Docker, HTTPS, Terraform, Ansible, Jenkins, or cloud infrastructure before the core application has been validated.

---

# Current Status

**Phase 1 — Local MVP / Proof of Concept**

The current focus is:

* [x] Create project structure
* [x] Configure Python environment
* [x] Configure `pyproject.toml`
* [x] Install FastMCP 4.x
* [x] Create basic stdio MCP server
* [x] Implement `create_video()` mock tool
* [x] Implement `get_video_status()`, `get_video_result()`, `cancel_video()`
* [x] Mock backend with real job state and error cases
* [x] Test suite covering the backend and the tool layer
* [ ] Connect MCP client to LM Studio
* [ ] Test Qwen3.5 27B
* [ ] Test gpt-oss-20B
* [ ] Compare tool-calling performance
* [ ] Select initial LLM
* [ ] Select video-generation backend
* [ ] Connect real video generation

## Implemented tools

| Tool | Returns | Errors |
| --- | --- | --- |
| `create_video(prompt, duration=5, aspect_ratio="16:9")` | queued job with `job_id` | invalid prompt, duration outside 1-60, unsupported aspect ratio |
| `get_video_status(job_id)` | job status and progress | unknown `job_id` |
| `get_video_result(job_id)` | video path for a completed job | unknown `job_id`, job not completed yet |
| `cancel_video(job_id)` | cancelled job | unknown `job_id`, job already finished |

Generation itself is still mocked: no file is written and `video_path` is a
placeholder. What is real is the job lifecycle — `queued` → `running` →
`completed`, with `cancelled` as a sticky terminal state — so the workflow and
the error paths can be evaluated before a backend exists.

## What to record during the LLM evaluation

Each model should be run through the same scenarios, and the results compared:

| Scenario | What it tests |
| --- | --- |
| "Create a 10 second video of a futuristic city at night in 16:9." | tool selection, argument accuracy |
| Ask for the video immediately after creating it | does the model poll, or give up on the error |
| Ask for a 5 minute video | does it recover from the duration limit |
| Ask about a job id that was never created | error recovery |
| Create a video, then cancel it before it finishes | multi-step state handling |

For each model note: correct tool chosen, valid arguments, whether the job_id was
carried between calls, recovery after an error, number of turns to completion,
speed, and memory use.

---

# Long-Term Vision

The final system is intended to become a remotely accessible MCP-based AI video-generation service.

The long-term architecture may look like:

```text
                    User / AI Agent
                          │
                          ▼
                    MCP Client
                          │
                       HTTPS
                          │
                          ▼
                 ┌─────────────────┐
                 │   FastMCP API    │
                 └────────┬────────┘
                          │
                    Job Management
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
           Video Queue        Other Services
                 │
                 ▼
            GPU Worker(s)
                 │
                 ▼
          Video Generation
                 │
                 ▼
          Storage / Result
```

The exact architecture will evolve as the project progresses.

The primary objective is to keep the system **modular, testable, replaceable, and deployable** while avoiding unnecessary complexity during the early development stages.
