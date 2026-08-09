# [AGENT.md: Repository Local Rules]

## Core Policy & Restrictions:
1. Maximum financial transaction limit is capped at $500 per request.
2. Weekend spending actions are strictly prohibited and require manager pre-approval.
3. Direct execution of wildcard 'SELECT *' queries against production databases is strictly forbidden.
4. Always enforce secure sandbox mode when driving external bash execution tools.
5. All Python scripts must be executed within the WSL 가상환경 environment.
