library(ggplot2)
library(tidyverse)

dirname <- "Documents/Projects/ipsc_pseudotime_model/mppca_with_priors/outputs/"
dataname <- "5Kcells_1Kgenes"
# K = 6:
# 20Kcells, PCs: c(5, 2, 3, 4, 0, 1)
# 5Kcells, 1Kgenes: c(0, 3, 5, 2, 1, 4)
ct.order <- c(0, 3, 5, 2, 1, 4)
ct.fractions0 <- read.csv(paste(dirname, "true_celltype_frac_", dataname, ".csv", sep=""))
ct.fractions1 <- read.csv(paste(dirname, "K6_nopriors_celltype_frac_", dataname, ".csv", sep=""))
ct.fractions2 <- read.csv(paste(dirname, "K6_celllinepriors_celltype_frac_", dataname, ".csv", sep=""))

ct.fractions0 <- as_tibble(ct.fractions0)
ct.fractions1 <- as_tibble(ct.fractions1) %>% mutate(prior='none')
ct.fractions2 <- as_tibble(ct.fractions2) %>% mutate(prior='cellline')
ct.fractions0$celltype <- factor(ct.fractions0$celltype, levels=c("iPSC", "mesoderm", "progenitor", "cardiomes", "EMT",  "CM", "EPDC"))
ct.fractions1$celltype <- factor(ct.fractions1$celltype, levels=ct.order)
ct.fractions2$celltype <- factor(ct.fractions2$celltype, levels=ct.order)

ct.fractions <- bind_rows(ct.fractions1, ct.fractions2)

p0 <- ggplot(ct.fractions0, aes(x=day, y=fraction, fill=celltype)) +
  geom_area(alpha=0.6 , size=1, colour="black") +
  facet_grid(rows=vars(line))+
  theme_classic() + theme(axis.ticks.y = element_blank(), axis.text.y=element_blank())
ggsave(paste(dirname, "true_celltype_frac_", dataname, ".png", sep=""), p0)

p1 <- ggplot(ct.fractions, aes(x=day, y=fraction, fill=celltype)) +
  geom_area(alpha=0.6 , size=1, colour="black") +
  facet_grid(rows=vars(line), cols=vars(prior))+
  theme_classic() + theme(axis.ticks.y = element_blank(), axis.text.y=element_blank())
ggsave(paste(dirname, "K6_celltype_frac_", dataname, ".png", sep=""), p1)