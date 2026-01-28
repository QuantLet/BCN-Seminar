"use client";
import React from "react";
import { useWeb3 } from "../../context/Web3Context";
import { useUI } from "../../context/UIContext";

export default function RewardsPage() {
  const {
    userRewards,
    username,
    account,
    isConnected,
    isAdmin,
    rounds,
    tokenSymbol,
    addTokenToMetaMask,
  } = useWeb3();
  const { showToast } = useUI();
  const handleAddToWallet = async () => {
    try {
      await addTokenToMetaMask();
      showToast("Token Added to MetaMask!", "success");
    } catch (e) {
      showToast("Failed to add token", "error");
    }
  };

  if (!isConnected)
    return (
      <div className="p-20 text-center text-slate-500">
        Please Connect Wallet
      </div>
    );

  // --- ADMIN VIEW: TOTAL DISTRIBUTIONS ---
  if (isAdmin) {
    // Calculate total tokens from rounds that have ended
    const endedRounds = rounds.filter((r) => !r.isActive);
    const totalDistributed = endedRounds.reduce(
      (acc, r) => acc + parseFloat(r.totalRewardPool),
      0,
    );

    return (
      <div className="max-w-7xl mx-auto px-6 py-10 animate-in fade-in duration-500">
        {/* Admin Header Stats */}
        <div className="bg-gradient-to-r from-purple-900 to-slate-900 p-8 rounded-2xl border border-purple-500/30 mb-10 shadow-2xl">
          <h2 className="text-slate-300 font-bold uppercase tracking-widest text-xs mb-2">
            Protocol Total Distribution
          </h2>
          <div className="flex items-end gap-4">
            <span className="text-5xl font-black text-white">
              {totalDistributed.toLocaleString()}
            </span>
            <span className="text-2xl font-bold text-purple-400 mb-1">
              {tokenSymbol}
            </span>
          </div>
          <p className="text-slate-400 text-sm mt-4">
            Total rewards allocated across {endedRounds.length} completed
            rounds.
          </p>
        </div>

        {/* Admin Distribution Table */}
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-purple-400">
            history
          </span>
          Distribution Log
        </h3>
        <div className="bg-white dark:bg-surface-dark/40 rounded-xl border border-border-dark overflow-hidden">
          <table className="w-full text-left">
            <thead className="bg-white/5 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-6 py-4">Round ID</th>
                <th className="px-6 py-4">Round Title</th>
                <th className="px-6 py-4">Date Ended</th>
                <th className="px-6 py-4 text-right">Pool Distributed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              {endedRounds.length === 0 ? (
                <tr>
                  <td
                    colSpan={4}
                    className="px-6 py-8 text-center text-slate-500"
                  >
                    No rounds have ended yet.
                  </td>
                </tr>
              ) : (
                endedRounds.map((r) => (
                  <tr key={r.id} className="hover:bg-white/5 transition-colors">
                    <td className="px-6 py-4 font-mono text-slate-500">
                      #{r.id}
                    </td>
                    <td className="px-6 py-4 font-bold">{r.title}</td>
                    <td className="px-6 py-4 text-slate-400 text-sm">
                      {new Date(r.endTime * 1000).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-purple-400 font-bold">
                      {r.totalRewardPool} {tokenSymbol}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // --- USER VIEW: PERSONAL REWARDS (Existing Logic) ---
  const totalEarned = userRewards.reduce(
    (acc, r) => acc + parseFloat(r.amount),
    0,
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 animate-in fade-in duration-500">
      {/* Profile Header */}
      <div className="bg-surface-dark p-8 rounded-2xl border border-border-dark mb-10 flex flex-col md:flex-row items-center gap-6 shadow-xl">
        <div className="size-24 rounded-full bg-gradient-to-tr from-primary to-emerald-500 flex items-center justify-center text-3xl font-bold shadow-lg">
          {username ? username.charAt(0).toUpperCase() : "U"}
        </div>
        <div className="flex-1 text-center md:text-left">
          <h2 className="text-3xl font-bold text-white">
            {username || "Anonymous User"}
          </h2>
          <p className="text-slate-400 font-mono text-xs bg-black/30 inline-block px-2 py-1 rounded mt-1">
            {account}
          </p>
        </div>
        <div className="text-right">
          <p className="text-slate-500 uppercase text-xs font-bold tracking-widest">
            My Total Earnings
          </p>
          <p className="text-4xl font-black text-accent-gold mt-1">
            {totalEarned.toFixed(2)}{" "}
            <span className="text-lg text-white">{tokenSymbol}</span>
          </p>
        </div>
        <div>
          {/* NEW BUTTON HERE */}
          <button
            onClick={handleAddToWallet}
            className="text-xs font-bold bg-white/10 hover:bg-white/20 border border-white/20 px-3 py-2 rounded-lg flex items-center gap-2 ml-auto transition-colors"
          >
            <img
              src="https://upload.wikimedia.org/wikipedia/commons/3/36/MetaMask_Fox.svg"
              className="w-4 h-4"
              alt="MetaMask"
            />
            Add {tokenSymbol} to Wallet
          </button>
        </div>
      </div>

      {/* History Table */}
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <span className="material-symbols-outlined text-accent-gold">
          savings
        </span>
        Reward History
      </h3>
      <div className="bg-white dark:bg-surface-dark/40 rounded-xl border border-border-dark overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-white/5 text-xs uppercase text-slate-500">
            <tr>
              <th className="px-6 py-4">Round</th>
              <th className="px-6 py-4">Rank Achieved</th>
              <th className="px-6 py-4">Date</th>
              <th className="px-6 py-4 text-right">Amount</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {userRewards.length === 0 ? (
              <tr>
                <td
                  colSpan={4}
                  className="px-6 py-8 text-center text-slate-500"
                >
                  No rewards earned yet. Go vote in active rounds!
                </td>
              </tr>
            ) : (
              userRewards.map((r, i) => (
                <tr key={i} className="hover:bg-white/5 transition-colors">
                  <td className="px-6 py-4 font-bold">{r.roundTitle}</td>
                  <td className="px-6 py-4">
                    <span className="bg-yellow-500/10 text-yellow-500 px-2 py-1 rounded text-xs font-bold border border-yellow-500/20">
                      #{r.rank}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-400 text-sm">
                    {new Date(r.timestamp * 1000).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-emerald-400 font-bold">
                    + {r.amount}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
