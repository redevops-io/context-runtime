# Release 0.5.1

A patch on [0.5.0](release-0.5.0.md); if you are upgrading from 0.4.1, read those notes. No settings or interfaces change.

- **An answer that failed the generation check is no longer recorded as verified.** The outcome event marked every answer
  that was not an abstention as `verified`, including one scored unfaithful and sent up the escalation ladder. `verified`
  now means the answer passed. Learning records written by earlier versions can carry `verified` on answers that did not.
- **The compaction archive is private to the user.** It stores whole tool outputs and arguments, which can include
  credentials. An explicit archive directory was created `0755` and its files `0644` under the usual umask. A directory the
  archive creates is now `0700` and every file `0600`. **Existing directories and files keep their modes**: check and
  tighten them (`chmod -R go-rwx <archive directory>`).
- The geo demo image publishes again from a version tag. The `v0.5.0` tag published no image.

## Verification

The test suite passes on the released commit.
