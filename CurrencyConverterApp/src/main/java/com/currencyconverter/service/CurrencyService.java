package com.currencyconverter.service;

import com.currencyconverter.model.ConversionRequest;
import com.currencyconverter.model.CurrencyRate;

public class CurrencyService {
    private final ExchangeRateFetcher fetcher = new ExchangeRateFetcher();

    public double convert(ConversionRequest request) {
        CurrencyRate rate = fetcher.fetchRates(request.getBaseCurrency());
        if (rate != null && rate.getRates().containsKey(request.getTargetCurrency())) {
            return request.getAmount() * rate.getRates().get(request.getTargetCurrency());
        }
        return -1;
    }
}
