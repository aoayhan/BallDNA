# Legal, privacy, and accessibility

_Last updated: 7 October 2026_

Basketball DNA is an independent, non-commercial portfolio and research demonstration. It is not affiliated with, endorsed by, or sponsored by the NBA, its teams, its players, DARKO, nbarapm, Basketball Reference, or any other data provider mentioned below.

## Privacy notice

Basketball DNA does not provide accounts, accept uploads, run advertising, or install its own analytics or tracking cookies. The application code does not attempt to identify visitors and does not persist player searches or comparison choices to a user database. Interface choices are held in the browser URL or the temporary Streamlit session needed to operate the app.

The app is hosted by Streamlit Community Cloud. Streamlit may process technical data such as IP addresses, request metadata, logs, app analytics, and cookies or local storage required to operate its service. Community Cloud is hosted in the United States. That processing is controlled by Streamlit/Snowflake and is described in the [Streamlit privacy policy](https://streamlit.io/privacy-policy) and [Streamlit trust and security documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security).

Basketball DNA loads its interface assets locally except for player headshots requested from the NBA CDN. When a headshot is displayed, the visitor's browser contacts that CDN and may disclose ordinary request metadata such as IP address, browser information, and the referring page. Basketball DNA does not load Google Fonts, advertising SDKs, or third-party analytics. The locally stored Ko-fi button contacts Ko-fi only after a visitor chooses to open it. Other source and repository links likewise contact their destination only after a click.

No consent banner is shown because Basketball DNA intentionally uses no non-essential cookies or trackers of its own. Streamlit may still use functional session technology that is necessary to deliver the requested app. This notice must be reviewed before adding analytics, embedded media, advertising, authentication, or other third-party SDKs.

Privacy or accessibility questions may be submitted through the project's [GitHub issue tracker](https://github.com/aoayhan/BallDNA/issues). Do not include private or sensitive information in a public issue.

## Accessibility statement

Basketball DNA aims to follow WCAG 2.2 AA practices. The interface uses native keyboard-operable controls, visible focus indicators, a skip link, semantic headings and tables, text labels that do not rely on color alone, and reduced-motion preferences. Functional graphics have accessible names; decorative graphics are hidden from assistive technology.

Known limitation: this is a portfolio prototype, not a certified accessibility conformance claim. Streamlit-generated widgets and the complete screen-reader experience have not been independently audited across every browser and assistive-technology combination. If something blocks access, please report the page, control, browser, and assistive technology through the [issue tracker](https://github.com/aoayhan/BallDNA/issues).

## Terms and limitations

- Basketball DNA provides experimental similarity and impact context for education and demonstration. It is not professional scouting, employment, medical, financial, legal, gambling, fantasy-sports, or betting advice.
- Similarity scores describe modelled basketball tendencies. They are not probabilities, player-quality rankings, predictions, or statements of fact about a person.
- Defensive Player DNA is under development and should not be relied upon by itself. All outputs may contain missing data, source errors, modelling errors, or outdated information.
- The service is provided “as is” and “as available,” without warranties of accuracy, availability, fitness for a particular purpose, or non-infringement, to the extent permitted by applicable law.
- Do not attempt to disrupt the service, bypass access controls, extract secrets, or use automated traffic that materially degrades availability.

Nothing in this notice creates permission to reuse third-party names, statistics, images, trademarks, or datasets beyond the rights granted by their respective owners and licences.

## Data, software, and brand credits

- Historical box scores: [Eoin A Moore's Kaggle dataset](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores), whose publisher declares CC0-1.0. Basketball DNA does not independently warrant the publisher's rights in upstream material.
- Shot and event archive: [cdechoch/nba-data-archive](https://huggingface.co/datasets/cdechoch/nba-data-archive), whose dataset card declares Apache-2.0 and identifies its upstream sources.
- Impact context: DARKO Daily Plus-Minus by Kostya Medvedovsky, accessed through the historical export at [nbarapm](https://nbarapm.com/datasets/MetricHistory). DARKO values are credited, shown only as separate context, and are not model inputs.
- NBA names and statistics are used for identification and analysis. NBA.com is attributed as an upstream source. NBA and team names and marks belong to their respective owners.
- Current-player headshots are displayed from the NBA CDN for identification. They remain the property of their respective rightsholders; Basketball DNA claims no ownership, affiliation, or endorsement and will remove disputed material on request.
- The homepage Allen Iverson artwork was supplied by the project owner for this non-commercial portfolio. Basketball DNA claims no ownership, affiliation, or endorsement and will remove disputed material on request.
- Ko-fi button artwork is an official Ko-fi brand asset used to link to the project's Ko-fi page.
- Basketball DNA's interface uses system fonts and does not distribute a third-party web font.

Source provenance, retrieval dates, and additional dataset notices are recorded in `data/historical/metadata.json` and `data/nba_snapshot/LICENSE-NOTICE.md`. Public deployment does not itself establish a licence to redistribute a source; permission must be obtained or the affected asset removed whenever the source terms are unclear.
