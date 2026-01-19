package com.currencyconverter.service;

import com.currencyconverter.config.ApiConfig;
import com.currencyconverter.model.CurrencyRate;
import com.google.gson.Gson;

import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class ExchangeRateFetcher {
    public CurrencyRate fetchRates(String baseCurrency) {
        try {
            String urlStr = ApiConfig.API_URL + "?base=" + baseCurrency;
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            InputStreamReader reader = new InputStreamReader(conn.getInputStream());
            CurrencyRate rate = new Gson().fromJson(reader, CurrencyRate.class);
            reader.close();
            return rate;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
