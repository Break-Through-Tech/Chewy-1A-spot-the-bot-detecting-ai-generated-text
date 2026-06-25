---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The tech stack is primarily Python-based, utilizing libraries such as Keras and FastAPI, making it suitable for student proficiency. |
| Data Readiness | 🟢 | The HC3 dataset is readily accessible and under 1GB, minimizing preprocessing time and enabling a focus on modeling. |
| Resource Check | 🟢 | All required tools and libraries can be accessed via free-tier services like Google Colab, ensuring no hardware constraints for students. |

**Student Fit Score:** 8/10  
**Technical Depth Score:** 7/10  
**Overall Recommendation:** APPROVE

**Advisor Feedback Draft:**

The project leverages relevant methodologies in NLP and classifier development, but the complexity of measuring robustness against diverse paraphrasing could challenge students' analytical skills. It is vital to clearly define the scope of the robustness analysis to ensure appropriate student engagement.

---

# Spot the Bot: Detecting AI-Generated Text

**Company / Org:** Chewy.com  
**Challenge Advisor:** Rishabh Jain, rishab1300@gmail.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Chewy.com

Chewy.com is a leading online retailer in the pet supplies industry, focusing on delivering high-quality products and services for pet owners. Our commitment to customer satisfaction drives our innovative approach across various departments, including technology and customer support.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use a public corpus of paired human-written and AI-generated (ChatGPT) answers — the HC3 dataset — to build a classifier that distinguishes human-written from AI-generated text and measure how well it holds up when the AI text is lightly paraphrased to evade detection. This will help address the growing challenge of content authenticity and academic integrity, determining whether text was written by a person or a machine and reveal where automated AI-text detectors fail in the real world.

### Success Criteria
Strong classification performance (macro-F1 and accuracy) on the held-out test set, reported against a majority-class baseline with precision/recall and a confusion matrix. A distinctive success criterion is a robustness curve showing performance degradation when AI text is paraphrased.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Understanding | Explore dataset, handle missing values, document findings |
| **October** | Model Development | Train baseline model, experiment with approaches, iterate |
| **November** | Evaluation & Presentation | Finalize model, prepare presentation, document results |

**Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** HC3 dataset, available via Hugging Face  
**Format:** CSV/TSV, JSON  
**Size:** under 1gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- The HC3 dataset: a public corpus of paired human-written and AI-generated (ChatGPT) answers spanning everyday, medical, legal, financial, and psychology questions. Available via Hugging Face in CSV/TSV and JSON formats.
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, NLP

**Recommended Libraries:**
- Natural Language Processing (NLP)
- Supervised Machine Learning
- Deep Learning (TF-IDF, Logistic Regression, Feedforward Neural Network in Keras)
- FastAPI
- Streamlit
- Google Colab
- Hugging Face Spaces

**Evaluation Metrics:**
- Accuracy, Precision/Recall, macro-F1 score

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Link to an article or blog post about the problem domain]
- [Link to an industry report or case study]

**Technical Tutorials:**
- [Link to a free tutorial on the ML technique(s) involved]
- [Link to documentation for a key library or tool]

**Code Examples:**
- [Link to a relevant GitHub repo]
- [Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab, VS Code
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
