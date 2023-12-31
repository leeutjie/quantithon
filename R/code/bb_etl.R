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

# data <- read_csv("./data/quantathon_hannah.csv")
data <- read_csv("C:/Users/hannah.denobrega/OneDrive - Prescient/projects/quantithon/data/quantathon_hannah.csv")

data_tr <-
  data %>%
  left_join(ret, by = "date") %>%
  # select(-c(SPX_PX_LAST, LUATTRUU_PX_LAST)) %>%
  relocate(SPX_RETURN, .before = SPX_PX_BID) %>%
  pivot_longer(names(.)[-1]) %>%
  arrange(name, date) %>%
  group_by(name) %>%
  mutate(
   value = case_when(
      name %in% str_subset(name, "PX_LAST") ~ log(value),
      name %in% str_subset(name, "RETURN") ~ value - lag(value)
    )
  ) %>%
  ungroup() %>%
  pivot_wider() %>%
  write_rds("data.rds")

