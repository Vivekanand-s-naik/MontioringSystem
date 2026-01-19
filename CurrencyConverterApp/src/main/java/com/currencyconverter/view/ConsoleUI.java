package com.currencyconverter.view;

import com.currencyconverter.model.ConversionRequest;

import java.util.Scanner;

public class ConsoleUI {
    private final Scanner scanner = new Scanner(System.in);

    public ConversionRequest getUserInput() {
        System.out.print("Enter base currency (e.g., USD): ");
        String base = scanner.nextLine().toUpperCase();

        System.out.print("Enter target currency (e.g., EUR): ");
        String target = scanner.nextLine().toUpperCase();

        System.out.print("Enter amount to convert: ");
        double amount = scanner.nextDouble();

        return new ConversionRequest(base, target, amount);
    }

    public void displayResult(double convertedAmount, String targetCurrency) {
        if (convertedAmount >= 0) {
            System.out.printf("Converted Amount: %.2f %s%n", convertedAmount, targetCurrency);
        } else {
            System.out.println("Conversion failed. Please check your input or try again later.");
        }
    }
}
