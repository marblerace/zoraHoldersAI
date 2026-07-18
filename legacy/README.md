# legacy/

The original standalone scripts this project grew out of: ad-hoc Python that
pulled Zora token holder data from the block explorer and produced holder-growth
plots. They are kept for provenance only.

**The application does not import or run anything in this folder.** The live
data path is reimplemented in `indexer/` with retries, pagination, transactional
upserts, and an audited sync log. These files read stale generated CSVs and are
excluded from linting.

Safe to delete if you want a leaner repo.
