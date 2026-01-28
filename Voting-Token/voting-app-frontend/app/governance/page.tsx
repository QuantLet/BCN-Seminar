"use client";
import React, { useState } from "react";
import Link from "next/link";
import { useWeb3 } from "../../context/Web3Context";
import { useUI } from "../../context/UIContext";

export default function GovernancePage() {
  const {
    rounds,
    isAdmin,
    contract,
    refreshData,
    contractBalance,
    tokenSymbol,
    refillTreasury,
    isConnected,
  } = useWeb3();
  const { showLoader, hideLoader, showToast, confirmAction } = useUI();

  // State for Minting Input
  const [mintAmount, setMintAmount] = useState("");
  const [isMinting, setIsMinting] = useState(false);

  // --- ACCESS CONTROL START ---
  if (!isConnected) {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center p-10 text-center animate-in fade-in zoom-in-95 duration-500">
        <span className="material-symbols-outlined text-6xl text-slate-600 mb-4">
          wallet
        </span>
        <h2 className="text-2xl font-bold text-white mb-2">
          Wallet Connection Required
        </h2>
        <p className="text-slate-400 mb-6">
          Please connect your wallet to access the Governance Dashboard.
        </p>
      </div>
    );
  }

  // Prevent Non-Admins from seeing this page content
  if (!isAdmin) {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center p-10 text-center animate-in fade-in zoom-in-95 duration-500">
        <span className="material-symbols-outlined text-6xl text-red-500 mb-4">
          lock_person
        </span>
        <h2 className="text-3xl font-bold text-white mb-2">Access Denied</h2>
        <p className="text-slate-400 mb-8 max-w-md">
          This dashboard is restricted to the Contract Owner. You do not have
          permission to view or manage governance rounds.
        </p>
        <Link
          href="/"
          className="bg-surface-dark border border-border-dark px-8 py-4 rounded-xl font-bold hover:bg-white/5 transition flex items-center gap-2"
        >
          <span className="material-symbols-outlined text-sm">arrow_back</span>{" "}
          Return Home
        </Link>
      </div>
    );
  }
  // --- ACCESS CONTROL END ---

  // --- HANDLERS ---
  const handleEndRound = async (id: number) => {
    if (!contract) return;

    // 1. Custom Confirmation Popup
    const confirmed = await confirmAction(
      "Are you sure you want to end this round? This will calculate votes, determine winners, and distribute tokens immediately."
    );
    if (!confirmed) return;

    try {
      // 2. Show Loader
      showLoader("Ending Round & Distributing Rewards...");

      const tx = await contract.endRound(id);
      await tx.wait();

      // 3. Success Toast
      showToast("Round Ended Successfully!", "success");
      refreshData();
    } catch (e: any) {
      console.error(e);
      // 4. Error Toast
      // Handle standard insufficient funds error nicely
      if (e.message && e.message.includes("Insufficient contract balance")) {
        showToast(
          "Error: Treasury empty! Please refill tokens first.",
          "error"
        );
      } else {
        showToast(e.reason || "Transaction Failed", "error");
      }
    } finally {
      // 5. Hide Loader
      hideLoader();
    }
  };

  const handleMint = async () => {
    if (!mintAmount) return;

    // Safety check for negative/zero amounts
    if (parseFloat(mintAmount) <= 0) {
      showToast("Please enter a valid positive amount.", "info");
      return;
    }

    try {
      setIsMinting(true);
      showLoader("Minting Tokens to Treasury...");

      await refillTreasury(mintAmount);

      setMintAmount("");
      showToast(`Successfully minted ${mintAmount} ${tokenSymbol}`, "success");
    } catch (e: any) {
      showToast(e.message || "Mint Failed", "error");
    } finally {
      setIsMinting(false);
      hideLoader();
    }
  };

  // Stats Calculations
  const activeRoundsCount = rounds.filter((r) => r.isActive).length;
  const totalPool = rounds.reduce(
    (acc, r) => acc + parseFloat(r.totalRewardPool),
    0
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* 1. TREASURY SECTION */}
      <div className="mb-12 bg-gradient-to-r from-slate-900 to-slate-800 border border-slate-700 p-8 rounded-3xl shadow-xl relative overflow-hidden group hover:border-slate-600 transition-colors">
        {/* Decorative Background */}
        <div className="absolute top-0 right-0 p-10 opacity-5 group-hover:opacity-10 transition-opacity duration-700">
          <span className="material-symbols-outlined text-9xl text-accent-gold">
            account_balance
          </span>
        </div>

        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <span className="material-symbols-outlined text-accent-gold">
            token
          </span>
          Treasury Management
        </h2>

        <div className="flex flex-col md:flex-row items-center gap-10">
          {/* Balance Display */}
          <div className="flex-1 w-full">
            <p className="text-slate-400 text-sm font-bold uppercase tracking-widest mb-2">
              Available for Rounds
            </p>
            <p className="text-5xl font-black text-white">
              {parseFloat(contractBalance).toLocaleString()}{" "}
              <span className="text-2xl text-accent-gold">{tokenSymbol}</span>
            </p>
            <p className="text-xs text-slate-500 mt-2">
              Smart Contract Balance
            </p>
          </div>

          {/* Minting Form */}
          <div className="w-full md:w-auto bg-black/20 p-6 rounded-2xl border border-white/5 backdrop-blur-sm">
            <label className="text-sm font-bold text-slate-300 mb-2 block">
              Refill Reward Pool
            </label>
            <div className="flex gap-2">
              <input
                type="number"
                placeholder="Amount"
                className="bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-accent-gold outline-none w-full md:w-48 transition-colors"
                value={mintAmount}
                onChange={(e) => setMintAmount(e.target.value)}
              />
              <button
                onClick={handleMint}
                disabled={isMinting || !mintAmount}
                className="bg-accent-gold hover:bg-yellow-500 text-slate-900 font-bold px-4 py-2 rounded-lg transition disabled:opacity-50 shadow-lg shadow-yellow-500/20"
              >
                {isMinting ? "Minting..." : "Mint Tokens"}
              </button>
            </div>
            <p className="text-[10px] text-slate-500 mt-2 max-w-xs">
              * Mints new tokens directly to the contract address. Only Admin
              can perform this action.
            </p>
          </div>
        </div>
      </div>

      {/* 2. HEADER */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-extrabold tracking-tight">
            Round History
          </h2>
          <p className="text-sm text-slate-500">
            Manage voting rounds and distributions.
          </p>
        </div>
        <Link
          href="/governance/create"
          className="bg-primary hover:bg-primary/90 text-white px-6 py-3 rounded-xl font-bold text-sm flex items-center gap-2 shadow-lg shadow-primary/20 active:scale-95 w-fit transition-all"
        >
          <span className="material-symbols-outlined">add_circle</span>
          Create New Round
        </Link>
      </div>

      {/* 3. STATS GRID */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <StatCard
          label="Total Rounds"
          value={rounds.length.toString()}
          icon="analytics"
          color="text-primary"
        />
        <StatCard
          label="Active Rounds"
          value={activeRoundsCount.toString()}
          icon="how_to_vote"
          color="text-emerald-500"
        />
        <StatCard
          label="Total Distributed"
          value={`${totalPool.toFixed(2)} ${tokenSymbol}`}
          icon="history"
          color="text-purple-400"
        />
      </div>

      {/* 4. TABLE */}
      <div className="bg-white dark:bg-surface-dark/40 rounded-2xl border border-slate-200 dark:border-border-dark overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-surface-dark/60 text-slate-500 text-xs font-bold uppercase tracking-wider">
                <th className="px-6 py-4">ID</th>
                <th className="px-6 py-4">Title</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4">Pool</th>
                <th className="px-6 py-4">Ends At</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-border-dark">
              {rounds.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-8 text-slate-500">
                    No rounds found on blockchain.
                  </td>
                </tr>
              ) : (
                rounds.map((round) => (
                  <tr
                    key={round.id}
                    className="hover:bg-slate-50/50 dark:hover:bg-surface-dark/20 transition-colors"
                  >
                    <td className="px-6 py-4 text-xs font-mono text-slate-500">
                      #{round.id}
                    </td>
                    <td className="px-6 py-4 font-bold">{round.title}</td>
                    <td className="px-6 py-4">
                      <StatusBadge isActive={round.isActive} />
                    </td>
                    <td className="px-6 py-4 font-mono text-accent-gold">
                      {round.totalRewardPool} {tokenSymbol}
                    </td>
                    <td className="px-6 py-4 text-xs text-slate-500">
                      {new Date(round.endTime * 1000).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {round.isActive && (
                        <button
                          onClick={() => handleEndRound(round.id)}
                          className="px-3 py-1.5 text-[10px] font-bold rounded-lg bg-red-500/10 text-red-500 hover:bg-red-500/20 border border-red-500/20 transition-colors cursor-pointer"
                        >
                          End Round
                        </button>
                      )}
                      {!round.isActive && (
                        <span className="text-xs text-slate-400 font-medium px-2 py-1 bg-slate-800 rounded">
                          Ended
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// Sub-components
const StatCard = ({ label, value, icon, color }: any) => (
  <div className="bg-surface-dark border border-border-dark p-6 rounded-2xl flex flex-col justify-between hover:border-slate-600 transition-colors">
    <div className="flex justify-between items-start">
      <span className="text-slate-500 text-sm font-medium">{label}</span>
      <span className={`material-symbols-outlined ${color}`}>{icon}</span>
    </div>
    <div className="mt-4">
      <h3 className="text-2xl font-bold">{value}</h3>
    </div>
  </div>
);

const StatusBadge = ({ isActive }: { isActive: boolean }) => (
  <span
    className={`px-2.5 py-1 rounded-full text-[10px] font-bold border uppercase ${
      isActive
        ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
        : "bg-red-500/10 text-red-500 border-red-500/20"
    }`}
  >
    {isActive ? "Active" : "Ended"}
  </span>
);
