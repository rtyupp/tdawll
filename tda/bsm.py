# -*- coding: utf-8 -*-
"""
Black-Scholes-Merton: الجريكس والتسعير.

الجزء `BSMCalculator.calculate` مقتبس من OSBT (https://github.com/DrEMG/osbt, MIT License,
Copyright (c) 2026 DrEMG) مع تعديلات صغيرة؛ وأضفتُ التسعير `price` ونسخة numpy لغاما الشبكة
(تُستخدم لحساب مستوى انقلاب الغاما "gamma flip" عند أسعار افتراضية).
"""
import math

import numpy as np


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


class BSMCalculator:
    """كل الدوال ساكنة وبلا مضاعف عقد (×100 يُطبّق خارجياً)."""

    @staticmethod
    def calculate(S, K, T, r, sigma, option_type):
        """delta (-1..1)، gamma (>0)، theta_per_day ($/سهم/يوم، سالب للشراء)، vega_per_pct ($/سهم/1% IV)."""
        zero = {"delta": 0.0, "gamma": 0.0, "theta_per_day": 0.0, "vega_per_pct": 0.0}
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return zero
        try:
            sq = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
            d2 = d1 - sq * sigma
            n1, N1, N2 = _pdf(d1), _cdf(d1), _cdf(d2)
            gamma = n1 / (S * sigma * sq)
            vega_pct = S * n1 * sq * 0.01
            if option_type == "call":
                delta = N1
                theta = -(S * n1 * sigma) / (2 * sq) - r * K * math.exp(-r * T) * N2
            else:
                delta = N1 - 1
                theta = -(S * n1 * sigma) / (2 * sq) + r * K * math.exp(-r * T) * _cdf(-d2)
            return {"delta": delta, "gamma": gamma, "theta_per_day": theta / 365.0, "vega_per_pct": vega_pct}
        except (ValueError, ZeroDivisionError, OverflowError):
            return zero

    @staticmethod
    def price(S, K, T, r, sigma, option_type):
        """سعر BSM للسهم الواحد. عند الانتهاء أو IV صفر يرجع القيمة الجوهرية."""
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return intrinsic
        try:
            sq = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
            d2 = d1 - sigma * sq
            if option_type == "call":
                return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
            return K * math.exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)
        except (ValueError, ZeroDivisionError, OverflowError):
            return intrinsic


_erfc = np.vectorize(math.erfc, otypes=[float])


def gamma_grid(S, K, T, sigma, r=0.045):
    """غاما BSM لمصفوفة عقود عند سعر S (مجمّع numpy). K,T,sigma مصفوفات بنفس الطول."""
    K, T, sigma = np.asarray(K, float), np.asarray(T, float), np.asarray(sigma, float)
    ok = (T > 0) & (sigma > 0) & (K > 0) & (S > 0)
    g = np.zeros_like(K)
    if not ok.any():
        return g
    sq = np.sqrt(T[ok])
    d1 = (np.log(S / K[ok]) + (r + 0.5 * sigma[ok] ** 2) * T[ok]) / (sigma[ok] * sq)
    g[ok] = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sigma[ok] * sq)
    return g


def norm_cdf_arr(x):
    return 0.5 * _erfc(-np.asarray(x, float) / math.sqrt(2))
