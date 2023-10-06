# Stratification

# generic function to compare performance:

Perf_comparisons <- function(Idxs, YMs, Alias){
  # For stepping through uncomment:
  # YMs <- Hi_Vol
  Unconditional_SD <-

    Idxs %>%

    group_by(Tickers) %>%

    mutate(Full_SD = sd(Return) * sqrt(252)) %>%

    filter(YearMonth %in% YMs) %>%

    summarise(SD = sd(Return) * sqrt(252), across(.cols = starts_with("Full"), .fns = max)) %>%

    arrange(desc(SD)) %>% mutate(Period = Alias) %>%

    group_by(Tickers) %>%

    mutate(Ratio = SD / Full_SD)

  Unconditional_SD

}
