# homework8
Haotian Chen

New York University

##Summary

In this project, I run an LDA topic modeling analysis for Amazon Fine Food Reviews (available from: https://www.kaggle.com/snap/amazon-fine-food-reviews). I used the R package lda and I visualize the output using LDAvis.

The R Code is available at: https://github.com/haotianchen93/homework8/blob/master/Homework%208.R

The LDA topic models analysis for top 10 relavant topics is available at: https://haotianchen93.github.io/homework8/vis

##Result Analysis

Topic 5 & 7 have large overlaps, because these customer reviews focused on information of taste and flavor. Topic 2 & 6 have some overlaps, mostly focused on various dimensions of differnet kinds of food. Topic 1 & 3 also overlap, and these two focused on the service Amazon provided, such as box, time, and price. Topic 4/8/9/10 are isolated, primarily because these four topics focused on different food: sugar, pet food, tea and coffee respectively. So we can see that these reviews are very comprehensive and varied.


##The data

First, I manually download the review .csv file to my mac and load the .csv file in R studio.

```r
#Topic Modeling: Amazon Fine Food Reviews

library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

#load csv file
precorpus<- read.csv("/Users/chenhaotian/Desktop/amazon reviews.csv", header=TRUE, stringsAsFactors=FALSE)

#passing Full Text to variable review
review <- precorpus$Text
```

##Pre-processing

Before fitting a topic model, I need to tokenize the text and remove all the punctuations and spaces. In particular, we use the English stop words from the "SMART" and some customized stop words.

```r
#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "said", "the", "also", "say", "just", "like","for", 
                "us", "can", "may", "now", "year", "according", "mr", "br", "www", "http")
stop_words <- tolower(stop_words)


review <- gsub("'", "", review) # remove apostrophes
review <- gsub("[[:punct:]]", " ", review)  # replace punctuation with space
review <- gsub("[[:cntrl:]]", " ", review)  # replace control characters with space
review <- gsub("^[[:space:]]+", "", review) # remove whitespace at beginning of documents
review <- gsub("[[:space:]]+$", "", review) # remove whitespace at end of documents
review <- gsub("[^a-zA-Z -]", " ", review) # allows only letters
review <- tolower(review)  # force to lowercase

## get rid of blank docs
review <- review[review != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(review, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)
```

##Using R package 'lda' for model fitting

The object document is a large list where each element represents one document. After creating this list, we compute a few statistics about the corpus, such as length and vocabulary counts:

```r
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 
```

Next, we set up a topic model with 10 topics, relatively diffuse priors for the topic-term distributions ($\eta$ = 0.02) and document-topic distributions ($\alpha$ = 0.02), and we set the collapsed Gibbs sampler to run for 3,000 iterations (slightly conservative to ensure convergence). A visual inspection of fit$log.likelihood shows that the MCMC algorithm has converged after 3,000 iterations. This block of code takes about 50 seconds to run on a Mac using a 2.4GHz i7 processor and 8GB RAM.

```r
# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  
```

##Visualizing the fitted model with LDAvis

To visualize the result using LDAvis, we will need estimates of the document-topic distributions, which we denote by the $D \times K$ matrix $\theta$, and the set of topic-term distributions, which we denote by the $K \times W$ matrix $\phi$. We estimate the "smoothed" versions of these distributions ("smoothed" means that we've incorporated the effects of the priors into the estimates) by cross-tabulating the latent topic assignments from the last iteration of the collapsed Gibbs sampler with the documents and the terms, respectively, and then adding pseudocounts according to the priors. A better estimator might average over multiple iterations of the Gibbs sampler (after convergence, assuming that the MCMC is sampling within a local mode and there is no label switching occurring), but we won't worry about that for now.

We've already computed the number of tokens per document and the frequency of the terms across the entire corpus. We save these, along with $\phi$, $\theta$, and vocab, in a list as the data object reviews.LDA.

```r
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

review_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
```

Now we're ready to call the createJSON() function in LDAvis. This function will return a character string representing a JSON object used to populate the visualization. The createJSON() function computes topic frequencies, inter-topic distances, and projects topics onto a two-dimensional plane to represent their similarity to each other. It also loops through a grid of values of a tuning parameter, $0 \leq \lambda \leq 1$, that controls how the terms are ranked for each topic, where terms are listed in decreasing of relevance, where the relevance of term $w$ to topic $t$ is defined as $\lambda \times p(w \mid t) + (1 - \lambda) \times p(w \mid t)/p(w)$. Values of $\lambda$ near 1 give high relevance rankings to frequent terms within a given topic, whereas values of $\lambda$ near zero give high relevance rankings to exclusive terms within a topic. The set of all terms which are ranked among the top-R most relevant terms for each topic are pre-computed by the createJSON() function and sent to the browser to be interactively visualized using D3 as part of the JSON object.

```r
library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = review_for_LDA$phi, 
                   theta = review_for_LDA$theta, 
                   doc.length = review_for_LDA$doc.length, 
                   vocab = review_for_LDA$vocab, 
                   term.frequency = review_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)
```

The serVis() function can take json and serve the result in a variety of ways. Here we'll write json to a file within the 'vis' directory (along with other HTML and JavaScript required to render the page). You can see the result at: https://dduwill.github.io/Product-Review/vis.
