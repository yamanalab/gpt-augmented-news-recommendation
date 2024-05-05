<h1 align="center"> News Recommendation with <br /> Category Description via Large Language Model </h1>


## Overview

<div align="center">
    <img src="./.github/images/method-overview.png" alt="Overview of our proposed method.">
</div>

This repository is the official implementation for the paper: **News Recommendation with Category Description via Large Language Model**. 

Personalized news recommendations are essential for online news platforms to assist users in discovering news articles that match their interests from a vast amount of online content. Prior research has revealed that content-based news recommendation models utilizing deep neural networks have demonstrated high performance.

News category information (e.g., ***tv-golden-glove***) serves as a crucial source of information for understanding news article content in news recommendation systems. However, news category names are too concise and often provide insufficient information. In this study, we propose a novel approach that utilizes Large Language Models (LLMs) to automatically generate descriptive texts for news categories, which are then applied to enhance news recommendation performance. Comprehensive experiments demonstrate that our proposed method achieves a 5.8% improvement in performance compared to baselines.