pimR::tool()

ret <-
  pimR::fetch(c("SPX Index", "LUATTRUU Index"),
              "RT116",
              min = F) %>%
  mutate(
    entity = stringr::str_replace(entity, " Index", ""),
    item = "RETURN"
  ) %>%
  unite(item, c(entity, item)) %>%
  select(date = datestamp, item, value) %>%
  pivot_wider(names_from = "item")

data <- read_csv("../data/quantathon_hannah.csv")

data %>%
  left_join(ret, by = "date") %>%
  select(-c(SPX_PX_LAST, LUATTRUU_PX_LAST)) %>%
  relocate(SPX_RETURN, .before = SPX_PX_BID) %>%
  write_rds("data.rds")
