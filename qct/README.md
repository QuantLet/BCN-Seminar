<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Title: pqc_demo.py — Post-Quantum Cryptography Risk Demonstration

Purpose: This script is a pedagogical demonstration of how post-quantum attacks (analogous to Shor’s algorithm) threaten elliptic-curve–based blockchain systems once public keys are revealed on-chain or in the mempool.

It illustrates, end-to-end, how: 1) Funds are locked to an address derived from a hash of a public key. 2) Spending a transaction reveals the public key and signature. 3) An attacker capable of solving the elliptic curve discrete logarithm can recover the private key from the public key. 4) The attacker can forge a competing transaction and steal funds via a mempool race.

Scope and Limitations: 
- This is NOT Bitcoin, Ethereum, or production cryptography.
- All elliptic curve parameters are intentionally tiny.
- The signature scheme is a toy Schnorr-like construction used solely to demonstrate “forge after key recovery.”
- Classical brute force is used to simulate the effect of Shor’s algorithm.

What This Demo Is: 
- A conceptual and educational model
- A presentation-friendly illustration of PQC risk
- A tool for explaining why public-key exposure matters

What This Demo Is NOT: 
- Secure cryptographic code
- A blockchain implementation
- A realistic performance or threat model

Key Insight: The cryptographic breaking point in many ECC-based systems is not address creation, but public-key revelation at spend time. Once ECC discrete log is broken, signatures become forgeable and transaction authenticity collapses.

Recommended Use: 
- Educational demos
- Risk discussions for post-quantum migration
- Security architecture presentations

Run: python pqc_demo.py

License / Disclaimer: For educational and illustrative purposes only. Do not use any part of this code in production systems.

```
