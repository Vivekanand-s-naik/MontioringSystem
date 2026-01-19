package com.currencyconverter.model;

public class ConversionRequest {
    private String baseCurrency;
    private String targetCurrency;
    private double amount;

    public ConversionRequest(String baseCurrency, String targetCurrency, double amount) {
        this.baseCurrency = baseCurrency;
        this.targetCurrency = targetCurrency;
        this.amount = amount;
    }

    public String getBaseCurrency() {
        return baseCurrency;
    }

    public String getTargetCurrency() {
        return targetCurrency;
    }

    public double getAmount() {
        return amount;
    }
}
