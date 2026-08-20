# Trace Link

Multi-agent pipeline trace:

https://smith.langchain.com/public/34272861-731c-4d97-85a5-a32eb7ad9d02/r/6bf74ff8-6ee1-4534-8769-668216506731

## Query

"Explain long-term memory architectures for AI agents"

## Structure

Root run `multi_agent_workflow` (35.41s) with the Supervisor's routing order nested
underneath it as child runs:

| Step | Duration |
|---|---:|
| researcher.run | 5.51s |
| analyst.run | 8.18s |
| writer.run | 16.70s |
| critic.run | 3.87s |

Citation coverage for this run (from `critic.run`'s attributes): 100%.
