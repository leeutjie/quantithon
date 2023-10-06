
# Data format:

#  date         Tickers     Return       YearMonth       Top         Bot
# |2014-05-09   |NED      | -0.0023556   |2014May       | 0.0515057  | -0.0454018|
# |2016-07-28   |SLM      | -0.0220366   |2016July      | 0.0477523  | -0.0496343|
# |2016-07-29   |SLM      | -0.0091211   |2016July      | 0.0477523  | -0.0496343|

GO_GARCH <- function(Idxs_L){
    # make data tbl_xts
    xts_rtn <- Idxs_H %>% tbl_xts(., cols_to_xts = "Return", spread_by = "Tickers")

    # A) Univariate GARCH specifications:
    uspec <- ugarchspec(
        variance.model = list(model = "gjrGARCH",
                              garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(1,
                                        0), include.mean = TRUE),
        distribution.model = "sstd")

    # use gjrgarch spec (uspec) from earlier
    multi_univ_garch_spec <- multispec(replicate(ncol(xts_rtn), uspec))

    # ------------------------ Step 2: The specs are now saved.
    # Build our GO_GARCH model

    # From first step individual garch definitions (GJR):

    # Go-garch model specification:
    spec.go <- gogarchspec(multi_univ_garch_spec,
                           distribution.model = 'mvnorm', # or manig.
                           ica = 'fastica') # Note: we use the fastICA

    cl <- makePSOCKcluster(10)

    multf <- multifit(multi_univ_garch_spec, xts_rtn, cluster = cl)

    fit.gogarch <- gogarchfit(spec.go,
                              data = xts_rtn,
                              solver = 'hybrid',
                              cluster = cl,
                              gfun = 'tanh',
                              maxiter1 = 40000,
                              epsilon = 1e-08,
                              rseed = 100)

    # print(fit.gogarch)



    # ------------------------ Step 3: Extracttime-varying conditional correlations:
    gog.time.var.cor <- rcor(fit.gogarch)
    gog.time.var.cor <- aperm(gog.time.var.cor,c(3,2,1))
    dim(gog.time.var.cor) <- c(nrow(gog.time.var.cor), ncol(gog.time.var.cor)^2)



    # ------------------------ Step 4: Rename to put G0-GAECH into a useable format
    gog.time.var.cor <- renamingdcc(ReturnSeries = xts_rtn, DCC.TV.Cor = gog.time.var.cor)

    gog.time.var.cor
}
