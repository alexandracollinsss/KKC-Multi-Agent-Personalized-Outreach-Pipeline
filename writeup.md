# KKC Outreach Pipeline Writeup

## Architecture decisions and tradeoffs

### Two-agent separation

The pipeline splits into two independent modules with a hard file boundary between them.

Agent 1 in `agent1_researcher.py` starts from a name, title, organization, and optional LinkedIn URL. It runs live web searches, concurrently fetches the most relevant pages, and passes a compact retrieval packet to Gemini. The model produces structured JSON that includes a draft email, a research sources list, and a claims array where every factual sentence in the email maps to a reachable URL with a confidence score. If it cannot find a specific sourced hook (a real interview, post, or panel appearance) the draft email is set to null rather than producing a generic one.

Agent 2 in `agent2_validator.py` loads that JSON file from disk. It strips all internal metadata keys before passing anything to the model, so it has no visibility into how Agent 1 reasoned or what it considered. It re-fetches every source URL independently, runs its own searches to cross-check title and contact fit, and applies LLM judgment on top of a deterministic rules layer.

The deterministic layer is what keeps validation honest. The LLM catches most problems but can miss things. Python code catches the rest such as claims without a source URL are always blocked, confidence below 0.3 is always blocked, confidence between 0.3 and 0.6 is always flagged for review, and source URLs that returned an HTTP 4xx or 5xx status are flagged regardless of what the model said. The LLM and the rules run independently and their outputs merge, keeping the strongest severity when both flag the same claim.

### Model choice

The pipeline defaults to `gemini-flash-latest`. The test runs that produced the output in this repository used `GEMINI_MODEL=gemini-flash-latest` with a fallback chain of `gemini-3-flash-preview,gemini-3.1-flash-lite-preview,gemini-2.5-flash,gemini-2.0-flash`. The agent output files record `gemini-3.1-flash-lite-preview` as the model that executed, which is where the chain settled during that run.

`gemini-flash-latest` is a good practical choice for this pipeline. It handles long context well, which matters because Agent 1 can pass a retrieval packet that runs into the hundreds of thousands of characters. The Flash tier is also meaningfully faster than Pro, which matters when a single contact run can take several minutes. The fallback chain means a rate limit or 503 on the primary model rolls over automatically rather than failing the run. Gemini was also chosen for its higher free-tier API limits, which made iterating without cost friction practical.

Both agents use the same provider. The independence requirement is enforced by the file boundary, not by using different APIs.

### Live retrieval

The pipeline does not use training data for facts about real people. Agent 1 runs roughly 40 search queries across base research, site-restricted queries against the org's own domain, and a third-party expansion pass if the initial sources are thin. Both searches and URL fetches run concurrently via thread pools.

Search routes through Serper (Google Search via serper.dev) when `SERPER_API_KEY` is set, falling back to `ddg_search` which tries DuckDuckGo Lite first and Bing HTML scraping second, then Jina Reader search as a last resort. Serper is what was used for the test runs in this repo. Jina Reader also serves as a fallback for individual page fetches where direct HTTP requests are blocked or the page is JavaScript-walled.

Agent 2 independently re-fetches every URL from Agent 1's claims list. A deterministic pass in `validation_rules.py` checks every source URL's HTTP status code and blocks the claim if the server returned a 4xx response, independent of the LLM's judgment.

### Fetch cache

URL responses can be cached to disk under `output/cache/` to speed up iterative runs on the same contacts. It is off by default and enabled with `FETCH_CACHE_ENABLED=1`. The cache skips failed or empty responses so stale 403s do not pin themselves across reruns.

### Results from test runs

Three contacts are discussed here. The fourth was excluded per the submission instructions.

**Sarah Guo, Conviction — WARN.** Sarah has a strong public footprint and the pipeline found it. The email opening references her appearance on Invest Like the Best, grounded in a verifiable Apple Podcasts URL. Two additional claims were verified against a Medium essay she wrote on venture-backable startups and her personal site. The WARN came from claim c4, which stated that Conviction focuses on healthcare and enterprise automation. The source was a product launch page for a single portfolio company. The validator correctly identified that one example does not constitute a firm-wide thesis and flagged it as an overstatement. This is the validator functioning correctly: surfacing the one place where the email went slightly beyond its source rather than silently passing it.

**Manny Maceda, Bain & Company — WARN.** The pipeline surfaced a CNBC-TV18 interview from WEF Davos 2026 where Maceda discussed CEOs shifting from AI pilot projects to deployment at scale. The claim verified cleanly against the source. His title as Chairman was confirmed independently via Wikipedia. The WARN came from a contact fit flag: the validator correctly identified that the Chairman of a global consulting firm is not a typical target for pre-seed VC outreach. What the system could not know is that Manny backs Kyber Knight directly, which makes the contact fit flag a false positive in this case. The tool has no awareness of the sender's existing relationships or network, so it evaluates every contact as if the outreach is cold. The email was approved with the caveat visible because the factual content is solid and the decision to send is a human judgment call, not a data quality problem.

**Laurene Powell Jobs, Emerson Collective — FAIL.** No LinkedIn URL was provided. The pipeline found her Milken Institute speaker page and an annual letter from the Emerson Collective site, confirming her title but not surfacing a specific hook. There was no interview quote, no recent panel statement, nothing concrete enough to open an email with. The spec requires a null draft in that case. The FAIL is the correct outcome: an email that opens with a vague reference to her public presence would be worse than no email, and the pipeline refused to produce one. The contact fit flag also fired correctly given the scale difference between a pre-seed fund and Emerson Collective.

## What I would build next with another 4 hours

The remaining bottleneck after search and fetch parallelism is Gemini latency on large prompts. The retrieval packet can reach several hundred thousand characters and a thinking-tier model on that size takes two to five minutes per agent. Caching fetched pages eliminates most of that on reruns, but the first run for a new contact is unavoidably slow. The most direct fix would be tightening the retrieval cap so the model receives a smaller, better-ranked packet rather than everything that was found.

A second priority would be relationship context. The Manny Maceda result demonstrated the clearest gap that the validator flagged him as a wrong contact because it has no visibility into KKC's existing network. Adding a lightweight relationship layer where the sender can mark contacts as warm or note a prior connection would let Agent 2 evaluate contact fit in the right frame rather than treating every outreach as cold.

The third priority is a more reliable LinkedIn access path. LinkedIn is the single highest-signal source for title verification and personal writing, and the current pipeline treats it as best-effort because automated requests get blocked or throttled. A browser-based approach with clear user opt-in, or an official partner data source, would make this reliable.

I would also expand the source types the pipeline actively searches. X posts and threads are often where people express views most directly, and those views are exactly the kind of specific hook the opening email needs. A third-party API providing reliable search access to X content, with the same claim-to-source traceability rules applied, would increase hook quality for active posters.

## Known failure modes in my current implementation

**No relationship context.** The validator evaluates every contact as a cold outreach target. It cannot distinguish between a new contact and someone who already has a relationship with the sender. This produced the Manny Maceda false positive. He is a legitimate warm contact flagged as misaligned because the system only sees his title and org, not his history with KKC.

**Low-profile contacts produce no output.** People with minimal public presence will not produce a sourced hook. The pipeline handles this correctly by refusing to draft an email rather than generating something generic, but it means the output rate is directly tied to how publicly active the contact is. Laurene Powell Jobs demonstrated this as she's very well-known, but her public writing did not surface a clean specific hook that connects to KKC's thesis at run time.

**Titles go stale.** Team pages update slowly and third-party sources lag further behind. The cross-reference between LinkedIn title and org website title catches mismatches that exist at run time, but a title confirmed today may be wrong in six months.

**Email inference is fragile.** The pipeline prefers leaving `resolved_email` null over guessing from domain conventions alone. This is the right tradeoff for accuracy but it means most contacts come through without a confirmed address.

**Pages behind walls reduce retrieval quality.** Paywalls, JavaScript rendering, and aggressive bot detection all reduce how much usable text gets into the retrieval packet. The Jina Reader fallback recovers some of these cases. When a contact's organization publishes most of its content behind a login, the research packet is thinner and the resulting email is more likely to WARN or FAIL validation.
