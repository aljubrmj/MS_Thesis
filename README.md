<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">MS Thesis (Only Snippets of Final Results)</h3>

  <p align="center">
  Smart completions enable physical measurements over space and time, which provides large volumes of information at unprecedented rates. However, the optimization of inflow control valve (ICV) settings of smart multilateral wells is a challenging task. Traditionally, ICV field tests, evaluating well performance at different ICV settings, are conducted to observe flow behavior and configure ICV’s, however this is often suboptimal. This study investigated a surrogate-based prediction and optimization algorithm that minimizes the number of ICV field tests required, predicts well performance of all unseen combination of ICV settings, and determines the optimal ICV setting and net present value (NPV).

To achieve the study objective and capture the extent and variation of the problem, five numerical reservoir models were considered: Case-A, Case-B, Case-C, Case-D, and Case-E. These models differ in many aspects, including geological settings, reservoir rock and fluid properties, number of fluid phases, etc. While four are synthetic models, Case-E is based on a real offshore field located in Saudi Arabia. To optimally determine candidate ICV field tests, several sampling techniques were investigated and applied, including random, space-filling, and adaptive sampling. Predictive surrogates were trained using cross-validated feedforward neural networks. Optimization was achieved using direct enumeration and mesh adaptive direct search (MADS). Both deterministic and stochastic optimization tasks were considered. Deterministic optimization tackles cases where operators are optimizing a real well drilled and completed in the field or optimizing an exact numerical reservoir model. Meanwhile, stochastic optimization is concerned with uncertain numerical reservoir models. The utility framework was utilized to account for decision makers' risk aversion along with the uncertainty and monetary value associated with different decisions of ICV settings.

 Algorithm performance was evaluated based on the number of ICV field tests required to: 1) surpass R\textsuperscript{2} thresholds of 80\% and 90\% on all unseen scenarios, and 2) match the optimal ICV settings and NPV. To determine the diminishing value of additional ICV field tests, the triangulation sampling loss was used as a stoppage criterion. Surface and downhole oil and water flow prediction and optimization were achieved successfully in the different reservoir models using this approach. Considering the real reservoir model of Case-E with multiple producers at the crest or periphery, the algorithm only required six and ten ICV field tests on average to achieve 80\% and 90\% R\textsuperscript{2} across the different scenarios of this real reservoir model. The algorithm achieved robust prediction results for different Case-E scenarios involving different well locations (crest and periphery) and wellbore configurations (forked and fishbone). Furthermore, uniform and Gaussian noise types were introduced at levels of 10\%, 30\%, and 50\% to the training production profiles to evaluate the model robustness to noise. The algorithm achieved similar results to the noise-free counterpart, hence it is robust to these common types of noise.  The resultant surrogate was also used to deterministically decide on the optimal settings of ICV devices and predict NPV effectively. Further improvement was accomplished through adaptively sampling and fitting the surrogate to rather predict NPV explicitly where NPV predictions were generated with nearly 95\% R\textsuperscript{2} given 20 ICV field tests.

Stochastic optimization was investigated by applying the utility framework coupled with MADS on the proposed surrogate-based prediction. With respect to risk aversion, different decision makers were considered (risk-averse, neutral, and risk-prone). The exponential utility was used as an analytical expression to evaluate the expected utility of a given decision of ICV settings. Stochastic optimization was demonstrated for a single-well scenario, where 10,240 simulation runs would be required to accomplish exhaustive analysis. Instead, the proposed algorithm achieved comparable optimal results with only 600 simulation runs, yielding 94\% saving in simulation time. Another scenario involved the stochastic optimization of two wells, where 5,242,880 simulation runs would be required to accomplish exhaustive analysis. Instead, the proposed algorithm achieved comparable optimal results with only 2,000 simulation runs, yielding 99.96\% saving in simulation time. 

Using adaptive sampling and machine learning proved effective in the prediction and optimization of surface and downhole flow profiles of smart wells. This method further allows for dynamically optimizing field strategy in a reinforcement learning setting where production data are used continuously to further improve the prediction performance.
    <br />
    <a href="https://github.com/aljubrmj/MS_Thesis"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/aljubrmj/MS_Thesis/issues">Report Bug</a>
    ·
    <a href="https://github.com/aljubrmj/MS_Thesis/issues">Request Feature</a>
  </p>
</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Name: [@MJAljubran](https://twitter.com/twitter_handle) - m.j.aljubran@gmail.com

Project Link: [https://github.com/aljubrmj/MS_Thesis](https://github.com/aljubrmj/MS_Thesis)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/aljubrmj/MS_Thesis.svg?style=for-the-badge
[contributors-url]: https://github.com/aljubrmj/MS_Thesis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aljubrmj/MS_Thesis.svg?style=for-the-badge
[forks-url]: https://github.com/aljubrmj/MS_Thesis/network/members
[stars-shield]: https://img.shields.io/github/stars/aljubrmj/MS_Thesis.svg?style=for-the-badge
[stars-url]: https://github.com/aljubrmj/MS_Thesis/stargazers
[issues-shield]: https://img.shields.io/github/issues/aljubrmj/MS_Thesis.svg?style=for-the-badge
[issues-url]: https://github.com/aljubrmj/MS_Thesis/issues
[license-shield]: https://img.shields.io/github/license/aljubrmj/MS_Thesis.svg?style=for-the-badge
[license-url]: https://github.com/aljubrmj/MS_Thesis/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mohammad-jabs/
[product-screenshot]: images/screenshot.png

