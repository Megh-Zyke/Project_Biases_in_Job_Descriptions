# Open questions

1. **Blocker: Potato cannot start on curry.** The base env has potato-annotation 2.3.0,
   which always writes generated templates to
   `/opt/anaconda/lib/python3.12/site-packages/potato/templates/generated/`
   (hard-coded in `flask_server.py`, `if config["site_dir"] == "default" or True`).
   That directory is owned by `anaconda:anaconda` (775) and the `anaconda` group has
   no members, so `potato start` fails with `PermissionError`. An admin needs to
   either create that `generated/` directory writable by the lab, or upgrade
   Potato (newer versions are what the skill documents). Upgrading also brings
   `potato validate` and CSV export.
2. Review the draft decision rules (`DRAFT_RULES` in `prepare.py`), especially Age
   (an explicit stated age counts as persona-originated) and Proximity (the
   lower-confidence tier).
3. Should evidence spans be mandatory? They are currently optional (see DESIGN.md).
4. Who logs in? `allow_all_users: true` is fine on curry. Set an allowlist before
   the server is reachable from outside the cluster.
