plotGOGARCH <- function(data){


    # ------------------------ Step 1: Specficiations to be used:
    # Using the rugarch package, specify univariate functions

    # A) Univariate GARCH specifications:
    uspec <- ugarchspec(
        variance.model = list(model = "gjrGARCH",
                              garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(1,
                                        0), include.mean = TRUE),
        distribution.model = "sstd")

    # B) Repeat uspec n times:
    multi_univ_garch_spec <- multispec(replicate(ncol(data), uspec))

    # ------------------------ Step 2: The specs are now saved.
    # Build our GO_GARCH model

    # From first step individual garch definitions (GJR):

    # Go-garch model specification:
    spec.go <- gogarchspec(multi_univ_garch_spec,
                           distribution.model = 'mvnorm', # or manig.
                           ica = 'fastica') # Note: we use the fastICA

    cl <- makePSOCKcluster(10)

    multf <- multifit(multi_univ_garch_spec, data, cluster = cl)

    fit.gogarch <- gogarchfit(spec.go,
                              data = data,
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
    gog.time.var.cor <- renamingdcc(ReturnSeries = data, DCC.TV.Cor = gog.time.var.cor)

    # ------------------------ Step 5: Plot

    # Rand <- ggplot(gog.time.var.cor %>% filter(grepl("ZAR_", Pairs), !grepl("_ZAR", Pairs))) +
    Rand <- ggplot(gog.time.var.cor %>% filter(grepl("SouthAfrica_", Pairs), !grepl("_SouthAfrica", Pairs))) +
    # Rand <- ggplot(gog.time.var.cor %>% filter(grepl("SouthAfrica_", Pairs), !grepl("_SouthAfrica", Pairs))) +
        geom_line(aes(x = date, y = Rho, colour = Pairs), size = 0.2) +
        theme_hc() +
        ggtitle("GARCH correlations to ZAR") +
        theme(legend.text=element_text(size=6), legend.title = element_blank())
    Rand

}

