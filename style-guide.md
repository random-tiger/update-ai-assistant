# On Updates: A Writing Style Guide  
**POC:** Blake Bassett  
**Last updated:** Sep 2, 2024  

## 1. Purpose  
This document provides guidance for writing inputs for the Weekly Tech Org Update.

## 2. Terms  
**“The What”**  
The core fact or result that you're reporting. This is the central piece of information you want to convey.  
*For example:* X personalized recommendations feature boosts user retention from 60% to 75% over the last quarter.  

**“The Why”**  
The reason behind the result or fact. This explains why the “What” happened or was important.  
*For example:* …because it better aligns content with user preferences.  

**“The So-What?”**  
The implication or significance of the result. This explains the impact of the “What” and “Why” on the business and the objective(s) we’re seeking to achieve.  
*For example:* …This improvement is projected to contribute an additional $1.2 million in revenue over the next fiscal year.

## 3. Weekly Updates  

### a. Metrics  
Lead with the name of the metric you moved beginning with the Platform(s) lifts and then Global lift in parentheses (). Note that for ML experiments, Global and Platform are the same thing. Format numbers with a plus sign (+) and two decimal places, followed by a percentage sign (%).  
- **This:** X experiment resulted in TVT +3.75% (1.2% Global).  
- **Not this:** X experiment resulted in 3% TVT lift  

Use the following metric name format:  

| Metric | Recommended |
|--------|--------------|
| conversion-5min | Conversion |
| registration | Registration |
| conversion-5min-new_visitors_d1 | D1 Conversion |
| visit_days | Visit Days |
| tvt | TVT |
| qualified_view_days | QVD |

### b. Titles and Headings  
Should be no more than 8 words. Must include a verb (action word) and capture the “So-What?”. First word should be capitalized; subsequent non-nouns should not be capitalized.  
- **This:** Mature age-gate garners uptick in new registrations.  
- **Not this:** A Significant Increase In New User Registrations Was Observed Due To The Implementation Of The Age-Gate Feature.  

### c. Dates  
Use M/D for dates.  
- **This:** 9/2  
- **Not this:** 09/02/2024 nor 09/02  

### d. Highlight Updates  
- **What it is:** Notable concluded experiments / launches with meaningful platform or topline impact.  
- **What it is not:** Planned or running experiments; project or program updates; completed tasks; experiments without meaningful platform or topline impact.  

These are not highlights:  
- Set to launch X experiment next week…  
- Ongoing experiment is showing X results…  
- Conducted a planning meeting on X project…

**Sentence structure:**  
**<bolded descriptive title with a period.> <The What>. <The Why>. <The So-What?>. <link>.**  
- **This:** Watchlist feature increased user retention. The new Watchlist feature led to +1.23% Retention, confirming our strategy to enhance user experience by making it easier to save and revisit content.  
- **Not this:** We released the Watchlist feature last week, and it is expected to improve user retention. Ongoing analysis will determine its impact.

### e. Flag Updates  
- **What it is:** Risks, blockers, or challenges that could negatively impact the project, timeline, or overall success. These are issues that require attention, decision-making, or mitigation strategies.  
- **What it is not:** Routine updates, minor inconveniences, ongoing work without significant risk, or issues that have already been resolved or are under control.

**Sentence structure:**  
**<Bolded descriptive title with a period.> <The What>. <The Why>. <The So-What?>. <The proposed action or next steps> <link>.**  
- **This:** AWS CDN contract delay risks H1 goal. The delay in finalizing the AWS contract for the new content delivery network (CDN) could postpone the rollout of the X feature by up to four weeks, affecting our H1 target. We have escalated the issue with the legal team and expect a resolution by the end of this week.  
- **Not this:** Contract delay. The vendor contract is taking longer than expected, which might delay the project. The team is handling it.

### f. Next Updates  
- **What it is:** Upcoming / planned (within 2 weeks) or ongoing experiments / launches.  
- **What it is not:** General plans or ideas without a specific timeline or items that are far in the future and not immediately actionable.

**Sentence structure:**  
**<Bolded action item with a period.> <The What>. <The Why / intended outcome or purpose> <The So-What?>. <ETA M/D>. <link>.**  
- **This:** Deep learning cold start experiment. X deep learning-based ranking model will be rolled out to 20% of users next week to evaluate its impact on content recommendations and user engagement.  
- **Not this:** Planning meeting for deep learning ranker. We plan to host an experiment planning meeting for X experiment next week.

## 4. General Guidance  

### a. Consider your audience  
When writing updates, always consider your audience. Your primary readers are stakeholders who need clear, concise, and actionable information. Your writing should be precise, easy to follow, and directly tied to the goals and outcomes of the project.

### b. Brevity  
Never use two words when one will do. Use adverbs sparingly.  
- **This:** The update improved app load time by 40%.  
- **Not this:** The recent software update significantly and dramatically improved overall app load time and speed by a notable 40%.

### c. Active voice  
The subject of the sentence should perform the action. Active voice structure = Subject + Verb + Object.  
- **This:** The product team (subject) analyzes (verb) user data (object).  
- **Not this:** User data (object) is analyzed (form of "to be" + past participle) by the product team (subject).

### d. Commas  
Use the Oxford / serial comma in series.  
- **This:** We offer movies, TV shows, and documentaries.  
- **Not this:** We offer movies, TV shows and documentaries.

### e. Conjunctions  
Place a comma before a conjunction when the words on both sides could be standalone sentences; omit the comma when they cannot.  
- **This:** We need to review the analytics, and we should prepare a report.  
- **Not this:** We need to review the analytics and we should prepare a report.  
- **This:** The team discussed the project and decided to proceed.  
- **Not this:** The team discussed the project, and decided to proceed.

### f. Parallelism  
Use the same pattern of words or structure in a series or list within a sentence. It ensures that your writing is balanced and easy to read, making comparisons or lists clear and effective.  
- **This:** The new feature improves user engagement, increases retention, and boosts overall satisfaction.  
- **Not this:** The new feature improves user engagement, increases retention, and is boosting overall satisfaction.

### g. Sentence length  
Sentences should include no more than 30 words; the fewer the better.  
- **This:** The product team reduced app load time by 13% by optimizing the codebase and implementing a more efficient caching strategy.  
- **Not this:** The product team was able to significantly reduce the app load time by thoroughly optimizing the existing codebase, implementing a more efficient caching strategy, and refining several backend processes.

### h. Prepositions  
Do not end a sentence with a preposition.  
- **This:** The new feature includes options that users can easily navigate through during onboarding.  
- **Not this:** The new feature includes options that users can easily navigate through.

## 5. Appendix  

### Progress to Goal Format  

| # | Workstream | Progress Against Goal |
|---|------------|-----------------------|
| 1 | Personalization & Recommendation ([link]) | Help users discover relevant content to watch. Capped TVT +0.79% (15.8% relative progress to goal) – behind +5% H1 goal Conversion +0.31% (15.5% relative progress to goal) – behind +2% H1 goal |

### Workstream Updates Format  

| # | Workstream ([link]) | Owner | Top 5 Sub-workstreams | Status | Updates |
|---|---------------------|-------|-----------------------|--------|---------|
| 1 | Personalization & Recommendation ([link]) | Reve Groendyke | 1. Core ranking and retrieval (hyper personalized returning user deep learning models) | On-Track | Progress against goal: <Goal metric> +#% – <behind, on-track, ahead> +#% H1 goal Highlights: <Bolded descriptive title with a period.> <What>. <Why>. <So-What?>. <link> Flags: <Bolded descriptive title with a period.> <The What>. <The Why>. <The So-What?>. <The proposed action or next steps>. <link> Next Up: <Bolded action item with a period.> <The What>. <The Why / intended outcome or purpose> <The So-What?>. <ETA M/D>. <link>. |
