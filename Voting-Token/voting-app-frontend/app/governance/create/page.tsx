"use client";
import React, { useState, useEffect } from "react";
import { useWeb3 } from "../../../context/Web3Context";
import { ethers } from "ethers";
import { useUI } from "../../../context/UIContext";
import Link from "next/link";

export default function CreateRoundPage() {
  // We use 'rounds' instead of 'activeRound' to support selecting from multiple active rounds
  const { contract, refreshData, isConnected, isAdmin, rounds } = useWeb3();
  const { showLoader, hideLoader, showToast } = useUI();

  // --- State: Create Round ---
  const [title, setTitle] = useState("");
  const [rewardAmount, setRewardAmount] = useState("");
  const [durationHours, setDurationHours] = useState("24"); // Default 24 hours

  // --- State: Add Candidate ---
  const [selectedRoundId, setSelectedRoundId] = useState<string>("");
  const [candName, setCandName] = useState("");
  const [candAddress, setCandAddress] = useState("");

  // Filter only active rounds for the dropdown
  const activeRounds = rounds.filter((r) => r.isActive);

  // Auto-select the most recent active round when data loads
  useEffect(() => {
    if (activeRounds.length > 0 && !selectedRoundId) {
      setSelectedRoundId(activeRounds[0].id.toString());
    }
  }, [rounds, selectedRoundId]);

  // --- Handler: Create Round ---
  const handleCreateRound = async () => {
    if (!contract) return;
    if (!title || !rewardAmount) {
      showToast("Please fill in Title and Reward Amount", "error");
      return;
    }

    try {
      showLoader("Creating Round on Blockchain...");

      // Convert hours to seconds
      const duration = Number(durationHours) * 3600;
      const reward = ethers.parseEther(rewardAmount);

      const tx = await contract.createRound(title, duration, reward);
      await tx.wait();

      showToast("Round Created Successfully!", "success");

      // Clear Form
      setTitle("");
      setRewardAmount("");

      await refreshData();
    } catch (e: any) {
      showToast(e.reason || e.message || "Creation Failed", "error");
    } finally {
      hideLoader();
    }
  };

  // --- Handler: Add Candidate ---
  const handleAddCandidate = async () => {
    if (!contract || !selectedRoundId) return;
    if (!candName || !candAddress) {
      showToast("Please fill in Name and Wallet Address", "error");
      return;
    }

    try {
      // Find title for clearer loader message
      const targetRound = activeRounds.find(
        (r) => r.id.toString() === selectedRoundId
      );
      const roundTitle = targetRound
        ? targetRound.title
        : `Round #${selectedRoundId}`;

      showLoader(`Adding ${candName} to ${roundTitle}...`);

      const tx = await contract.addCandidate(
        selectedRoundId,
        candName,
        candAddress
      );
      await tx.wait();

      showToast(`Candidate added to ${roundTitle}!`, "success");

      // Clear Form
      setCandName("");
      setCandAddress("");

      await refreshData();
    } catch (e: any) {
      showToast(e.reason || e.message || "Failed to add candidate", "error");
    } finally {
      hideLoader();
    }
  };

  // --- Access Control ---
  if (!isConnected)
    return (
      <div className="p-20 text-center text-slate-500">
        Please connect wallet.
      </div>
    );
  if (!isAdmin)
    return (
      <div className="p-20 text-center text-red-500 font-bold">
        ⛔ Access Denied: Admin Only.
      </div>
    );

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">Admin Operations</h1>
        <Link
          href="/governance"
          className="text-sm text-slate-400 hover:text-white transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      <div className="grid gap-8">
        {/* --- CARD 1: CREATE NEW ROUND --- */}
        <div className="bg-surface-dark p-8 rounded-2xl border border-border-dark shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <span className="flex items-center justify-center size-8 rounded-full bg-primary/20 text-primary font-bold">
              1
            </span>
            <h2 className="text-xl font-bold text-white">Start New Round</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase ml-1">
                Round Title
              </label>
              <input
                className="w-full p-3 mt-1 rounded-xl bg-black/20 border border-slate-700 focus:border-primary outline-none transition-colors text-white"
                placeholder="e.g. Best Developer 2024"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase ml-1">
                  Reward Pool
                </label>
                <input
                  className="w-full p-3 mt-1 rounded-xl bg-black/20 border border-slate-700 focus:border-primary outline-none transition-colors text-white"
                  placeholder="Tokens (e.g. 100)"
                  type="number"
                  value={rewardAmount}
                  onChange={(e) => setRewardAmount(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase ml-1">
                  Duration (Hours)
                </label>
                <input
                  className="w-full p-3 mt-1 rounded-xl bg-black/20 border border-slate-700 focus:border-primary outline-none transition-colors text-white"
                  type="number"
                  value={durationHours}
                  onChange={(e) => setDurationHours(e.target.value)}
                />
              </div>
            </div>

            <button
              onClick={handleCreateRound}
              disabled={!title || !rewardAmount}
              className="bg-primary hover:bg-primary/90 disabled:bg-slate-800 disabled:text-slate-500 text-white py-4 px-6 rounded-xl font-bold w-full transition shadow-lg shadow-primary/20 mt-2"
            >
              Create Round on Blockchain
            </button>
          </div>
        </div>

        {/* --- CARD 2: ADD CANDIDATES --- */}
        <div className="bg-surface-dark p-8 rounded-2xl border border-border-dark shadow-xl opacity-90">
          <div className="flex items-center gap-3 mb-6">
            <span className="flex items-center justify-center size-8 rounded-full bg-emerald-500/20 text-emerald-500 font-bold">
              2
            </span>
            <h2 className="text-xl font-bold text-white">Add Candidates</h2>
          </div>

          {activeRounds.length === 0 ? (
            <div className="p-6 bg-slate-800/50 rounded-xl text-center border border-dashed border-slate-700">
              <p className="text-slate-400">
                No active rounds found. Please create a round first.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Round Selector */}
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase ml-1">
                  Target Round
                </label>
                <select
                  className="w-full p-3 mt-1 rounded-xl bg-black/20 border border-slate-700 focus:border-emerald-500 outline-none text-white cursor-pointer"
                  value={selectedRoundId}
                  onChange={(e) => setSelectedRoundId(e.target.value)}
                >
                  {activeRounds.map((r) => (
                    <option key={r.id} value={r.id}>
                      #{r.id} - {r.title} (Pool: {r.totalRewardPool})
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <input
                    className="w-full p-3 rounded-xl bg-black/20 border border-slate-700 focus:border-emerald-500 outline-none text-white"
                    placeholder="Candidate Name"
                    value={candName}
                    onChange={(e) => setCandName(e.target.value)}
                  />
                </div>
                <div className="flex-[2]">
                  <input
                    className="w-full p-3 rounded-xl bg-black/20 border border-slate-700 focus:border-emerald-500 outline-none font-mono text-sm text-white"
                    placeholder="Wallet Address (0x...)"
                    value={candAddress}
                    onChange={(e) => setCandAddress(e.target.value)}
                  />
                </div>
              </div>

              <button
                onClick={handleAddCandidate}
                disabled={!selectedRoundId || !candName || !candAddress}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white py-4 px-6 rounded-xl font-bold w-full transition shadow-lg shadow-emerald-600/20"
              >
                Add Candidate to Selected Round
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
