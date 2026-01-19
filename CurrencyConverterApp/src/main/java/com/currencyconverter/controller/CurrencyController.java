package com.currencyconverter.controller;

import com.currencyconverter.model.ConversionRequest;
import com.currencyconverter.service.CurrencyService;
import com.currencyconverter.view.ConsoleUI;

public class CurrencyController {
    private final CurrencyService service = new CurrencyService();
    private final ConsoleUI ui = new ConsoleUI();

    public void start() {
        ConversionRequest request = ui.getUserInput();
        double result = service.convert(request);
        ui.displayResult(result, request.getTargetCurrency());
    }
}
