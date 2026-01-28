"use client";
import React, { useState, useMemo } from "react";
import { useWeb3 } from "../../context/Web3Context";

export default function LeaderboardPage() {
  const { rounds, allCandidates, loading } = useWeb3();

  const [selectedRoundId, setSelectedRoundId] = useState<number | null>(
    rounds.length > 0 ? rounds[0].id : null,
  );

  const currentRound = rounds.find((r) => r.id === selectedRoundId);
  const candidates = selectedRoundId
    ? allCandidates[selectedRoundId] || []
    : [];

  const sortedCandidates = useMemo(() => {
    return [...candidates].sort((a, b) => b.voteCount - a.voteCount);
  }, [candidates]);

  const top3 = sortedCandidates.slice(0, 3);
  const rest = sortedCandidates.slice(3);

  // --- HELPER: Determine Style based on Votes ---
  const getPodiumStyle = (candidate: any) => {
    if (!candidate) return {};

    const maxVotes = top3[0]?.voteCount || 0;
    const secondVotes = top3[1]?.voteCount || 0;

    // TIE FOR 1ST PLACE
    if (candidate.voteCount === maxVotes && maxVotes > 0) {
      return {
        height: "h-48",
        color: "bg-accent-gold",
        text: "text-accent-gold",
        rank: 1,
        shadow: "shadow-[0_0_30px_rgba(250,204,21,0.2)]",
        rankIcon: "emoji_events", // Trophy
      };
    }

    // TIE FOR 2ND PLACE (Or is 2nd Place)
    if (candidate.voteCount === secondVotes && secondVotes > 0) {
      return {
        height: "h-32",
        color: "bg-slate-500", // Silver-ish
        text: "text-slate-300",
        rank: 2,
        shadow: "shadow-xl",
        rankIcon: null,
      };
    }

    // 3RD PLACE
    return {
      height: "h-24",
      color: "bg-orange-700", // Bronze
      text: "text-orange-400",
      rank: 3,
      shadow: "shadow-xl",
      rankIcon: null,
    };
  };

  if (loading)
    return (
      <div className="p-10 text-center animate-pulse">
        Loading Leaderboard...
      </div>
    );

  return (
    <div className="animate-in fade-in duration-500 max-w-7xl mx-auto px-6 py-10">
      <header className="flex flex-col md:flex-row justify-between items-end mb-12 gap-6">
        <div>
          <h2 className="text-4xl font-bold mb-2">Leaderboard</h2>
          <p className="text-slate-400">
            View rankings for active and past rounds.
          </p>
        </div>

        <div className="w-full md:w-64">
          <label className="text-xs font-bold text-slate-500 uppercase mb-2 block">
            Select Round
          </label>
          <select
            className="w-full bg-surface-dark border border-border-dark rounded-xl px-4 py-3 outline-none focus:border-primary text-white cursor-pointer"
            value={selectedRoundId || ""}
            onChange={(e) => setSelectedRoundId(Number(e.target.value))}
          >
            {rounds.map((r) => (
              <option key={r.id} value={r.id}>
                #{r.id} - {r.title} ({r.isActive ? "Active" : "Ended"})
              </option>
            ))}
          </select>
        </div>
      </header>

      {!currentRound ? (
        <div className="text-center text-slate-500 py-10">
          No rounds available.
        </div>
      ) : (
        <>
          <div className="flex gap-4 mb-10 text-sm">
            <div className="px-4 py-2 bg-surface-dark rounded-lg border border-border-dark">
              Status:{" "}
              <span
                className={
                  currentRound.isActive
                    ? "text-green-500 font-bold"
                    : "text-red-500 font-bold"
                }
              >
                {currentRound.isActive ? "Live" : "Ended"}
              </span>
            </div>
            <div className="px-4 py-2 bg-surface-dark rounded-lg border border-border-dark">
              Pool:{" "}
              <span className="text-accent-gold font-bold">
                {currentRound.totalRewardPool}
              </span>
            </div>
          </div>

          {/* PODIUM SECTION */}
          {top3.length > 0 ? (
            <section className="flex justify-center items-end gap-4 md:gap-8 mb-16 h-64 border-b border-border-dark/50 pb-16">
              {/* --- LEFT COLUMN (Usually 2nd Place) --- */}
              <div className="flex flex-col items-center w-1/3 max-w-[150px]">
                {top3[1] && (
                  <>
                    <PodiumBar
                      candidate={top3[1]}
                      style={getPodiumStyle(top3[1])}
                    />
                  </>
                )}
              </div>

              {/* --- CENTER COLUMN (Always 1st Place) --- */}
              <div className="flex flex-col items-center w-1/3 max-w-[150px] z-10">
                {top3[0] && (
                  <>
                    <PodiumBar
                      candidate={top3[0]}
                      style={getPodiumStyle(top3[0])}
                      isCenter={true}
                    />
                  </>
                )}
              </div>

              {/* --- RIGHT COLUMN (Usually 3rd Place) --- */}
              <div className="flex flex-col items-center w-1/3 max-w-[150px]">
                {top3[2] && (
                  <>
                    <PodiumBar
                      candidate={top3[2]}
                      style={getPodiumStyle(top3[2])}
                    />
                  </>
                )}
              </div>
            </section>
          ) : (
            <div className="text-center py-10 text-slate-500">
              No votes cast yet.
            </div>
          )}

          {/* LIST VIEW */}
          <div className="space-y-3 max-w-4xl mx-auto">
            {rest.map((c, i) => (
              <div
                key={c.id}
                className="flex items-center justify-between p-4 bg-surface-dark border border-border-dark rounded-xl hover:border-primary/30 transition"
              >
                <div className="flex items-center gap-4">
                  <span className="font-mono text-slate-500 text-lg w-8 text-center">
                    #{i + 4}
                  </span>
                  <div>
                    <p className="font-bold">{c.name}</p>
                    <p className="text-xs text-slate-500">{c.wallet}</p>
                  </div>
                </div>
                <div className="font-mono font-bold text-lg">
                  {c.voteCount} Votes
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// --- SUB COMPONENT FOR CLEANER JSX ---
const PodiumBar = ({ candidate, style, isCenter }: any) => {
  return (
    <>
      <div className={`mb-2 text-center ${isCenter ? "-mt-8" : ""}`}>
        {style.rankIcon && (
          <span
            className={`material-symbols-outlined ${style.text} text-4xl block`}
          >
            {style.rankIcon}
          </span>
        )}
        <p
          className={`font-bold ${style.text === "text-accent-gold" ? "text-accent-gold" : "text-slate-300"} mb-1`}
        >
          {candidate.name}
        </p>
      </div>

      {/* Dynamic Height Bar */}
      <div
        className={`w-full ${style.height} ${style.color} rounded-t-xl flex items-end justify-center pb-2 ${style.shadow} transition-all duration-500`}
      >
        {/* <span className="text-4xl font-black text-black/30">{style.rank}</span> */}
      </div>

      <p className="mt-2 font-mono text-sm font-bold">
        {candidate.voteCount} Votes
      </p>
    </>
  );
};
