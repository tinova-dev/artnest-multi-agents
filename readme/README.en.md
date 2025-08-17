# 🎨 ArtNest - AI-powered Copyright Protection System

*"Protecting Creativity with AI"*  
A multi-agent based copyright protection system designed to help creators facing issues of plagiarism and unauthorized use of their artworks.

This project was developed as part of the [2025 Bolt Hackathon](https://worldslargesthackathon.devpost.com/).  
This repository contains the implementation of the **multi-agent system** of ArtNest.  👉 [Frontend Repository](https://github.com/tinovadev/artnest-frontend)

---

## Project Overview

### Background
Inspired by the real struggles of my sister, who majored in fine arts. **ArtNest** was built to leverage AI to prevent unauthorized training, detect plagiarism, and ensure traceability of digital artworks.

### Summary
- **Period:** May – June 2025  
- **Team**  
  - **Minha Sohn**: Multi-agent system (100%), frontend & backend (50%)  
  - **Kyungseo Park**: Algorand blockchain integration (100%), frontend & backend (50%)  
  - **Gawon Bae**: UI/UX Design  
- **Submission page & Demo video:** https://devpost.com/software/project-1qlz5jvgpocy  

---

## Key Features

### Web-based Image Crawling
- With a single click, users can trigger Puppeteer to automatically scroll through web pages.  
- The system collects visually similar images to the user’s work from the internet.  
- The multi-agent system then evaluates and analyzes the collected images for potential plagiarism.

### Copyright Protection
- **Invisible Watermarking**: Embeds imperceptible signals to discourage AI training on artworks.  
- **Adversarial Noise**: Perturbs the feature space of an image to hinder unauthorized model training.  

### Similarity Detection
- **CLIP**: Text–image similarity for style and composition comparison  
- **DISTS & LPIPS**: Structural and perceptual similarity metrics  
- **Grad-CAM**: Visualizes which regions of an image the model focuses on  

### Copyright Tracking & Verification
- Watermarks embedded in artworks allow source traceability.  
- When similar images are detected, the system verifies authorship and potential AI training usage.  


## ERD
*(To be added or described if diagrams are unavailable.)*


## Tech Stack

| Category           | Tools & Frameworks                         |
| ------------------ | ------------------------------------------- |
| Frontend / Backend | Next.js, TailwindCSS, ShadcnUI, Bolt.new    |
| Multi-Agent System | Google ADK, CrewAI                          |
| AI Models          | CLIP, DISTS, LPIPS, Grad-CAM                |
| DevOps             | Google Cloud (Cloud Run, GCS, PostgreSQL)   |


### Why These Tech Stack?

**Bolt.new Hackathon Constraints**  
- The hackathon required most of the development to be done within **Bolt.new**.  
- We chose **Next.js** for both frontend and backend development, since it integrates well with Bolt.new.  

**Google Cloud Hackathon Attempt**  
- Bolt.new only supports **JavaScript**, so Python-based multi-agent components had to be developed separately.  
- In parallel, we attempted the [Google Cloud Multi-Agent Hackathon](https://googlecloudmultiagents.devpost.com/) using $100 credits to build and deploy the multi-agent system with **Gemini, Cloud Run, GCS, and PostgreSQL**.  
- Due to limited time, we could not complete the submission, but gained valuable experience deploying in a cloud-native setup.  


## Reflections & Lessons Learned

### Limitations of Web Crawling
- Implemented Puppeteer-based automatic scrolling to collect candidate images, but repeated requests triggered **Google IP blocking**.  
- Tried multiple workarounds such as **headless mode, User-Agent modification, and request throttling**, but could not achieve stable results within the hackathon timeframe.  
- Learned that **production-grade solutions require official APIs, proxy rotation, and backoff strategies** instead of naive scraping.  

### Vulnerability of Copyright Marking
- Inserted **adversarial noise** into images to prevent AI models from learning, but GPT-based models **ignored the perturbations and still described the original artwork accurately**.  
- This confirmed that **simple noise injection is insufficient** as a defensive measure.  
- Learned that stronger methods such as **invisible watermarking, more advanced adversarial perturbations, and opt-out metadata standards** must be combined for effective protection.  

