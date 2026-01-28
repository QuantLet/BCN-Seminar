"use client";
import React, { useState } from "react";
import { useWeb3 } from "../../context/Web3Context";
import { useUI } from "../../context/UIContext";

export default function VotingPage() {
  const {
    rounds,
    allCandidates,
    userVoteStatus,
    contract,
    refreshData,
    isConnected,
    loading,
  } = useWeb3();
  const { showLoader, hideLoader, showToast, confirmAction } = useUI();

  const [votingId, setVotingId] = useState<number | null>(null);

  // Filter only active rounds
  const activeRounds = rounds.filter((r) => r.isActive);

  const handleVote = async (
    roundId: number,
    candidateId: number,
    candidateName: string
  ) => {
    if (!contract) return;

    // 1. Check if user already voted (Client-side pre-check)
    if (userVoteStatus[roundId]) {
      showToast("⚠️ You have already voted in this round!", "error");
      return;
    }

    const confirmed = await confirmAction(
      `Vote for ${candidateName} in this round?`
    );
    if (!confirmed) return;

    try {
      setVotingId(candidateId);
      showLoader("Casting Vote...");

      const tx = await contract.vote(roundId, candidateId);
      await tx.wait();

      showToast("Vote Cast Successfully!", "success");
      refreshData();
    } catch (err: any) {
      // 2. Handle Smart Contract Revert (Double protection)
      if (err.reason && err.reason.includes("Already voted")) {
        showToast("⚠️ Transaction Reverted: You already voted.", "error");
      } else {
        showToast(err.reason || "Vote Failed", "error");
      }
    } finally {
      setVotingId(null);
      hideLoader();
    }
  };

  if (!isConnected)
    return (
      <div className="p-20 text-center text-xl text-slate-500">
        Please Connect Wallet
      </div>
    );
  if (loading)
    return (
      <div className="p-20 text-center text-xl animate-pulse">
        Loading Rounds...
      </div>
    );
  if (activeRounds.length === 0)
    return (
      <div className="p-20 text-center text-xl text-slate-400">
        No Active Voting Rounds.
      </div>
    );

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-16">
      {/* Loop through ALL active rounds */}
      {activeRounds.map((round) => {
        const candidates = allCandidates[round.id] || [];
        const hasVoted = userVoteStatus[round.id];

        return (
          <div key={round.id} className="relative">
            {/* Round Header */}
            <div className="mb-8 pl-4 border-l-4 border-primary">
              <div className="flex items-center gap-4 mb-2">
                <h2 className="text-3xl md:text-5xl font-black">
                  {round.title}
                </h2>
                {hasVoted && (
                  <span className="bg-emerald-500 text-black px-3 py-1 rounded-full text-xs font-bold uppercase shadow-lg shadow-emerald-500/20">
                    <span className="material-symbols-outlined align-middle text-sm mr-1">
                      check_circle
                    </span>
                    Voted
                  </span>
                )}
              </div>
              <p className="text-slate-400">
                Pool:{" "}
                <span className="text-accent-gold font-bold">
                  {round.totalRewardPool} Tokens
                </span>
                <span className="mx-3">•</span>
                Ends: {new Date(round.endTime * 1000).toLocaleDateString()}
              </p>
            </div>

            {/* Candidates Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {candidates.length === 0 ? (
                <p className="col-span-3 text-slate-600 italic">
                  Candidates appearing soon...
                </p>
              ) : (
                candidates.map((c) => (
                  <div
                    key={c.id}
                    className={`p-6 rounded-2xl border transition group relative overflow-hidden ${
                      hasVoted
                        ? "bg-surface-dark/50 border-border-dark opacity-75"
                        : "bg-surface-dark border-border-dark hover:border-primary/50"
                    }`}
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-xl font-bold">{c.name}</h3>
                        <p className="text-xs font-mono text-slate-500">
                          {c.wallet.slice(0, 6)}...{c.wallet.slice(-4)}
                        </p>
                      </div>
                      <div className="bg-black/30 px-3 py-1 rounded text-xs text-slate-400 font-mono">
                        #{c.id}
                      </div>
                    </div>

                    <div className="flex justify-between items-end mt-6">
                      <div>
                        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">
                          Votes
                        </p>
                        <p className="text-2xl font-mono text-white font-bold">
                          {c.voteCount.toString()}
                        </p>
                      </div>

                      <button
                        onClick={() => handleVote(round.id, c.id, c.name)}
                        disabled={votingId !== null || hasVoted} // Disable if voted
                        className={`px-6 py-2 rounded-xl font-bold shadow-lg transition-all active:scale-95 flex items-center gap-2 ${
                          hasVoted
                            ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                            : "bg-primary hover:bg-primary/90 text-white shadow-primary/20"
                        }`}
                      >
                        {votingId === c.id ? "..." : hasVoted ? "Done" : "VOTE"}
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Divider */}
            <div className="w-full h-px bg-gradient-to-r from-transparent via-border-dark to-transparent mt-16"></div>
          </div>
        );
      })}
    </div>
  );
}
