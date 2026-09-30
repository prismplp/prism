:- module(data, []).
:- dynamic weather/2.
:- discontiguous weather/2.
weather(d1, sunny).
weather(d2, rainy).
weather(d3, snowy).
