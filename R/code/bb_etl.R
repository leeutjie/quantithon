pimR::tool()

ret <-
  pimR::fetch("SPX Index", "RT116", min=F) %>%
  select(date = datestamp, item, value) %>%
  mutate(item = "SPX_RETURN") %>%
  pivot_wider(names_from = "item")

data <- read_csv("../data/quantathon_hannah.csv")

data %>%
  left_join(ret, by = "date") %>%
  select(-SPX_PX_LAST) %>%
  relocate(SPX_RETURN, .before = SPX_PX_BID) %>%
  write_rds("data.rds")
