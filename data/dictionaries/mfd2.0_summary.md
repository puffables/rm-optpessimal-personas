**Moral Foundations Dictionaries for Linguistic Analyses, 2.0 (MFD
2.0)**

Recommended citation:

Frimer, J. A., Boghrati, R., Haidt, J., Graham, J., & Dehgani, M.
(2019). *Moral Foundations Dictionary for Linguistic Analyses 2.0*.
Unpublished manuscript.

**Abstract**

The original Moral Foundations Dictionaries (MFD) have unknown
psychometric properties, and fewer words in them (~32 per dictionary)
than many conventional linguistic analytic dictionaries. Our objective
was to test their validity and develop a larger and potentially more
valid set of MFDs. After generating large sets of candidate words, we
used word2vec software to estimate the prototypicality of each word
within their respective foundations. We then tested the validity of the
new dictionaries, and whether the validity diminished by trimming the
dictionaries to include only highly prototypic words. Results suggest
that the new dictionaries are more valid than the original set, and that
using only highly prototypic words did not diminish the dictionaries’
validity. We conclude by recommending that future research rely on the
full-length MFD 2.0.

**Background.**

The original MFDs have relatively few words (32 on average; see Table
1). A number of words that are prototypic to the foundations are absent
in the dictionaries. For instance, the care vice foundation (harm) does
not include the words *murder*, *torture*, or *agony*. Whether or not
these omissions limit the validity of the original dictionaries is
unknown. Our first goal was to test the validity of the original MFDs.

Our second goal was to develop a new set of MFDs with more words, and
test their validity too. Standard linguistic dictionaries in LIWC
typically have hundreds of words in each. This led us to wonder whether
increasing the number of words might improve the validity of the
dictionaries. On the other hand, some of Morteza’s recent work suggests
that after about 30 words, increasing the number of words in a
dictionary does not improve validity. We begin by developing new
dictionaries. We then test the validity of the new and original
dictionaries.

**Step 1. Word Lists Generation**

We generated enormous lists of words for each foundation. Jon and Jesse
then selected those that they thought were conceptually relevant to each
foundation. The result was 210 words per dictionary on average (see
Table 1).

**Table 1**. The number of words/word stems in the original MFD, and the
new MFD in its entirety, and with low, medium, and high prototypicality
inclusion criteria.

<table style="width:100%;">
<colgroup>
<col style="width: 18%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 9%" />
</colgroup>
<thead>
<tr>
<th><strong> </strong></th>
<th colspan="2" style="text-align: center;"><strong>Care</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Fairness</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Loyalty</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Authority</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Sanctity</strong></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong> </strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Average</strong></td>
</tr>
<tr>
<td>Original MFD</td>
<td style="text-align: right;">16</td>
<td style="text-align: right;">35</td>
<td style="text-align: right;">26</td>
<td style="text-align: right;">18</td>
<td style="text-align: right;">29</td>
<td style="text-align: right;">23</td>
<td style="text-align: right;">45</td>
<td style="text-align: right;">37</td>
<td style="text-align: right;">35</td>
<td style="text-align: right;">54</td>
<td style="text-align: right;"><strong>32</strong></td>
</tr>
<tr>
<td>New MFD – full dictionaries</td>
<td style="text-align: right;">182</td>
<td style="text-align: right;">288</td>
<td style="text-align: right;">115</td>
<td style="text-align: right;">236</td>
<td style="text-align: right;">142</td>
<td style="text-align: right;">49</td>
<td style="text-align: right;">301</td>
<td style="text-align: right;">130</td>
<td style="text-align: right;">272</td>
<td style="text-align: right;">388</td>
<td style="text-align: right;"><strong>210</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 0.4</td>
<td style="text-align: right;">72</td>
<td style="text-align: right;">110</td>
<td style="text-align: right;">54</td>
<td style="text-align: right;">115</td>
<td style="text-align: right;">56</td>
<td style="text-align: right;">34</td>
<td style="text-align: right;">118</td>
<td style="text-align: right;">75</td>
<td style="text-align: right;">130</td>
<td style="text-align: right;">160</td>
<td style="text-align: right;"><strong>92</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 0.8</td>
<td style="text-align: right;">46</td>
<td style="text-align: right;">62</td>
<td style="text-align: right;">34</td>
<td style="text-align: right;">66</td>
<td style="text-align: right;">28</td>
<td style="text-align: right;">28</td>
<td style="text-align: right;">71</td>
<td style="text-align: right;">46</td>
<td style="text-align: right;">72</td>
<td style="text-align: right;">96</td>
<td style="text-align: right;"><strong>55</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 1.2</td>
<td style="text-align: right;">32</td>
<td style="text-align: right;">40</td>
<td style="text-align: right;">25</td>
<td style="text-align: right;">30</td>
<td style="text-align: right;">17</td>
<td style="text-align: right;">15</td>
<td style="text-align: right;">39</td>
<td style="text-align: right;">20</td>
<td style="text-align: right;">36</td>
<td style="text-align: right;">51</td>
<td style="text-align: right;"><strong>31</strong></td>
</tr>
</tbody>
</table>

**  **

**Step 2. Prototypicality Estimation**

We wanted to know which words were most and least prototypic of each
foundation. To do so, we used word2vec software. It relies on the idea
that related words appear close to one another in text passages. For
example, the words *protect* and *safety* are likely to appear close to
one another in texts because they are conceptually related. Word2vec
uses a very large text corpus to estimate the prototypicality (“cosine”)
of target words to a set of seed words. Jon and Jesse selected seed
words for each foundation (see Table 2).

**Table 2.** Seed words for each foundation that were used to generate
prototypicality estimates.

<table>
<colgroup>
<col style="width: 12%" />
<col style="width: 18%" />
<col style="width: 14%" />
<col style="width: 18%" />
<col style="width: 17%" />
<col style="width: 18%" />
</colgroup>
<tbody>
<tr>
<td>Valence</td>
<td colspan="5" style="text-align: center;">Foundation</td>
</tr>
<tr>
<td></td>
<td style="text-align: center;">Care</td>
<td style="text-align: center;">Fairness</td>
<td style="text-align: center;">Loyalty</td>
<td style="text-align: center;">Authority</td>
<td style="text-align: center;">Sanctity</td>
</tr>
<tr>
<td>Virtue</td>
<td style="text-align: center;"><p>kindness</p>
<p>compassion</p>
<p>nurture</p>
<p>empathy</p></td>
<td style="text-align: center;"><p>fairness</p>
<p>equality</p>
<p>justice</p>
<p>rights</p></td>
<td style="text-align: center;"><p>loyal</p>
<p>team player</p>
<p>patriot</p>
<p>fidelity</p></td>
<td style="text-align: center;"><p>authority</p>
<p>obey</p>
<p>respect</p>
<p>tradition</p></td>
<td style="text-align: center;"><p>purity</p>
<p>sanctity</p>
<p>sacred</p>
<p>wholesome</p></td>
</tr>
<tr>
<td>Vice</td>
<td style="text-align: center;"><p>suffer</p>
<p>cruel</p>
<p>hurt</p>
<p>harm</p></td>
<td style="text-align: center;"><p>cheat</p>
<p>fraud</p>
<p>unfair</p>
<p>injustice</p></td>
<td style="text-align: center;"><p>betray</p>
<p>treason</p>
<p>disloyal</p>
<p>traitor</p></td>
<td style="text-align: center;"><p>subversion</p>
<p>disobey</p>
<p>disrespect</p>
<p>chaos</p></td>
<td style="text-align: center;"><p>impurity</p>
<p>depravity</p>
<p>degradation</p>
<p>unnatural</p></td>
</tr>
</tbody>
</table>

Morteza and Reihane then computed the prototypicality of each word in
each foundation. To establish baseline prototypicality numbers, we also
computed the prototypicality of 82 non-moral but common words (e.g.,
*drink*, *dawn*, *age*). As expected, the prototypicality estimates of
foundations words were much higher than non-moral words, across the
board (sparing you the details).

To create a common language across the foundations, we included both
foundation and non-moral words and computed z-scores of prototypicality
ratings within each foundation. For example, within the care virtue
foundation, the word *caring* was highly prototypic of the foundations
(z = 2.74), with words like *nurtures* (z = 1.25), *generously* (0.50),
*healthiness* (0.00), *relieve* (-0.47), and *consoles* (-1.16) being
successively less prototypic. We later use z-scores to set cut-offs for
creating smaller dictionaries with only highly prototypic words.

**Step 3. Validity Test**

The final objective was to test whether word density analyses with the
MFDs could successfully distinguish texts of known content. We asked
people from around the world to write essays about the foundations, then
tested how well the dictionaries picked out the content.

**Sample**. We recruited 1144 participants on the crowdsourcing website,
http://crowdflower.com. Crowdflower is similar to Amazon’s Mechanical
Turk except that Crowdflower has participants from many more countries.
Each participant received \$0.50 to write an essay about one of the 10
moral foundations. A research assistant read each essay and identified
ones that were not in English (104), respondents that declined the task
(34), and incoherent texts (25). Following standard Pennebakerian
protocols, we excluded an additional 256 responses that were less than
50 words long because short texts give unreliable word density
estimates.

The final sample, *N* = 656, was 37% female, 33 years old on average
(*SD* = 11) and from 58 different countries. The most common ones were
Venezuela (*n* = 79), Egypt (62), the US (58), Serbia (42), Ukraine
(39), India (30), Russia (29), Greece (21), Mexico (21), Italy (17),
Spain (16), Canada (16), Germany (14), Philippines (14), Turkey (12),
Argentina (12), Croatia (12), the UK (10), and Moldova (10). We asked
participants to indicate their political ideology on social issues on a
scale ranging from -100 (*extremely liberal*) to 100 (*extremely
conservative*). The average participant was slightly liberal (-7) but
the sample was quite diverse (*SD* = 50)

**Procedure**. We randomly assigned participants to write an essay about
one of the 10 foundations. For instance, the instructions for the care
virtue read:

> Please take a moment to recall a specific event in which a person
> (protagonist) acted with kindness, compassion, or empathy, or nurtured
> another person. The person who did this could have been you, or
> someone you know of. The person could also be a fictional individual
> from a book, movie, or TV show. When you have an event in mind, please
> proceed to the next page to answer some questions.

For the other foundations, the words in the first sentence after “person
(protagonist)” were replaced with foundation-specific prompts (see Table
3.) Participants wrote in response to three questions: (a) “What led up
to the event?” (b) “What did the person (protagonist) do?” and (c) “What
were the outcomes and consequences?” After each question, the
instructions were to “please explain thoroughly” before writing in three
successive text boxes. We combined all the text from each participant to
form short essays. Essays were 113 words long on average (*SD* = 100).

**Table 3.** Instructions for participants for the various foundations.

| **Foundation** | **Valence** | **Instructions** |
|----|----|----|
| Care | Virtue | …acted with kindness, compassion, or empathy, or nurtured another person. |
|  | Vice | …acted with cruelty, or hurt or harmed another person/animal and caused suffering. |
| Fairness | Virtue | …acted in a fair manner, promoting equality, justice, or rights.  |
|  | Vice | …was unfair or cheated, or caused an injustice or engaged in fraud.  |
| Loyalty | Virtue | …acted with fidelity, or as a team player, or was loyal or patriotic. |
|  | Vice | …acted disloyal, betrayed someone, was disloyal, or was a traitor. |
| Authority | Virtue | …obeyed, or acted with respect for authority or tradition. |
|  | Vice | …disobeyed or showed disrespect, or engaged in subversion or caused chaos. |
| Sanctity | Virtue | …acted in a way that was wholesome or sacred, or displayed purity or sanctity. |
|  | Vice | …was depraved, degrading, impure, or unnatural.  |

For example, a 59-year-old female from the Ukraine was assigned to the
*care virtue* condition. She wrote:

> During the Great Patriotic War, an acquaintance of my grandmother lost
> a family: his wife, son, parents. He came from the war as a hero, but
> he was alone and did not see the point in life. He adopted a child, an
> orphan who lost his parents in the same war. The child was 6 years
> old, he was homeless, hungry, unhappy. This man gave all his love,
> care for the baby. Two lonely hearts melted, warming each other. A
> veteran of the war had a meaning in life. He cared for the child,
> sprouted it, fed it, dressed it, taught it. The child acquired a
> caring father and very soon ceased to cry in a dream, experiencing all
> the horrors of the war. The kindness of the former warrior gave the
> child the opportunity to survive in a difficult post-war period. And
> the man has a new meaning in life.

We then used LIWC to estimate the density of words from each of the 10
foundations in each essay. Our analyses examined whether the density of
words in a foundation (e.g., care virtue) was higher than the density of
the same dictionary words in the other 9 foundations. To keep things
simple, we present the results as Cohen *d* effect sizes in Table 4. For
example, a *d* = +0.56 in the loyalty virtue foundation with the new,
full dictionary means that the density of loyalty virtue words was about
half a standard deviation higher in the loyalty virtue essays than all
other essays.

The first two rows of Table 4 show that the average validity of the new
dictionaries was higher (*d* = 0.36) than that of the original
dictionaries (*d* = 0.25). To test whether we benefit from having so
many words in the new dictionaries, we created shorter, tighter MFDs
with only highly prototypic words (using cut-offs of z \> 0.40, 0.80,
and 1.20, respectively; see Table 1 for dictionary word counts). Table 4
shows that the average validity neither increased nor decreased much as
a result. In sum, the new dictionaries are more valid than the old. And
trimming the new dictionaries to include only highly prototypic words
seems to have few costs.

**Table 4.** Validity of the various MFDs. Numbers represent Cohen’s d
effect sizes distinguishing the density of words corresponding to a
specific foundation (e.g., care virtue) to the density of the same
dictionary of words in all the other 9 foundations.

<table style="width:100%;">
<colgroup>
<col style="width: 18%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 9%" />
</colgroup>
<thead>
<tr>
<th><strong> </strong></th>
<th colspan="2" style="text-align: center;"><strong>Care</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Fairness</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Loyalty</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Authority</strong></th>
<th colspan="2"
style="text-align: center;"><strong>Sanctity</strong></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong> </strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Virtue</strong></td>
<td style="text-align: center;"><strong>Vice</strong></td>
<td style="text-align: center;"><strong>Average</strong></td>
</tr>
<tr>
<td>Original MFD</td>
<td style="text-align: right;">0.11</td>
<td style="text-align: right;">0.63</td>
<td style="text-align: right;">0.35</td>
<td style="text-align: right;">-0.03</td>
<td style="text-align: right;">-0.03</td>
<td style="text-align: right;">0.10</td>
<td style="text-align: right;">0.48</td>
<td style="text-align: right;">0.26</td>
<td style="text-align: right;">0.23</td>
<td style="text-align: right;">0.40</td>
<td style="text-align: right;"><strong>0.25</strong></td>
</tr>
<tr>
<td>New MFD – full dictionaries</td>
<td style="text-align: right;">0.51</td>
<td style="text-align: right;">0.37</td>
<td style="text-align: right;">0.19</td>
<td style="text-align: right;">0.58</td>
<td style="text-align: right;">0.42</td>
<td style="text-align: right;">0.56</td>
<td style="text-align: right;">0.20</td>
<td style="text-align: right;">0.11</td>
<td style="text-align: right;">0.42</td>
<td style="text-align: right;">0.26</td>
<td style="text-align: right;"><strong>0.36</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 0.4</td>
<td style="text-align: right;">0.34</td>
<td style="text-align: right;">0.38</td>
<td style="text-align: right;">0.20</td>
<td style="text-align: right;">0.62</td>
<td style="text-align: right;">0.42</td>
<td style="text-align: right;">0.55</td>
<td style="text-align: right;">0.27</td>
<td style="text-align: right;">0.19</td>
<td style="text-align: right;">0.41</td>
<td style="text-align: right;">0.37</td>
<td style="text-align: right;"><strong>0.38</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 0.8</td>
<td style="text-align: right;">0.14</td>
<td style="text-align: right;">0.52</td>
<td style="text-align: right;">0.31</td>
<td style="text-align: right;">0.68</td>
<td style="text-align: right;">0.17</td>
<td style="text-align: right;">0.55</td>
<td style="text-align: right;">0.39</td>
<td style="text-align: right;">0.26</td>
<td style="text-align: right;">0.49</td>
<td style="text-align: right;">0.24</td>
<td style="text-align: right;"><strong>0.37</strong></td>
</tr>
<tr>
<td>New MFD z &gt; 1.2</td>
<td style="text-align: right;">0.18</td>
<td style="text-align: right;">0.57</td>
<td style="text-align: right;">0.29</td>
<td style="text-align: right;">0.64</td>
<td style="text-align: right;">0.18</td>
<td style="text-align: right;">0.53</td>
<td style="text-align: right;">0.35</td>
<td style="text-align: right;">0.26</td>
<td style="text-align: right;">0.36</td>
<td style="text-align: right;">0.31</td>
<td style="text-align: right;"><strong>0.37</strong></td>
</tr>
</tbody>
</table>

Here are the same numbers represented graphically. The new dictionaries
are measurably better but by no means knocking it out of the park.

**Figure 1**. Validity of the various MFDs (Table 4 represented
graphically). Black dots represent the validity of a particular
foundation. Large red dots represent the average across all foundations.

**Conclusion**. We recommend that researchers use of the MFD 2.0.
Although the full-length version has no better or worse construct
validity than the shorter variants, we recommend the full-length version
because it more fully captures each foundation.
