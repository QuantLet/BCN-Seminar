"use client";
import Link from "next/link";
import { useWeb3 } from "../context/Web3Context";
import Image from "next/image";

export default function Home() {
  const {
    isConnected,
    rounds,
    activeRound,
    connectWallet, // Import the connect function
    tokenSymbol,
  } = useWeb3();

  // Simple calculated stats
  const totalRounds = rounds.length;
  const totalPool = rounds.reduce(
    (acc, r) => acc + parseFloat(r.totalRewardPool),
    0,
  );

  return (
    <div className="max-w-7xl mx-auto px-6 pt-10 animate-in fade-in duration-500">
      <section className="relative mb-20 py-20 text-center">
        {/* Dynamic Badge */}
        <div className="inline-block mb-4 px-4 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-wider shadow-[0_0_10px_rgba(59,130,246,0.2)]">
          {activeRound
            ? `● Live Round: ${activeRound.title}`
            : "Decentralized Governance System"}
        </div>

        {/* Hero Title */}
        <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-6 leading-[1.1]">
          Vote. Earn. <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-emerald-400">
            Govern.
          </span>
        </h1>

        <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
          Participate in on-chain governance rounds. Top candidates earn
          automatic rewards via Smart Contracts securely on the blockchain.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          {isConnected ? (
            <Link
              href="/explore"
              className="bg-primary hover:bg-primary/90 text-white px-8 py-4 rounded-xl font-bold text-lg transition shadow-lg shadow-primary/25 active:scale-95"
            >
              Vote Now
            </Link>
          ) : (
            <button
              onClick={connectWallet}
              className="bg-primary hover:bg-primary/90 text-white px-8 py-4 rounded-xl font-bold text-lg transition shadow-lg shadow-primary/25 active:scale-95 flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined">wallet</span>
              Connect Wallet
            </button>
          )}

          <Link
            href="/leaderboard"
            className="bg-surface-dark border border-border-dark hover:bg-white/5 px-8 py-4 rounded-xl font-bold text-lg transition active:scale-95"
          >
            View Leaderboard
          </Link>
        </div>
        <div className="inline-flex items-center gap-3 mt-6 border border-border-dark px-4 py-2 rounded-lg bg-surface-dark">
          <Image
            src="/images/img1.png"
            alt="Google Faucet"
            width={40}
            height={40}
          />
          <a
            href="https://cloud.google.com/application/web3/faucet/ethereum/sepolia"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary font-bold"
          >
            Google Faucet
          </a>
        </div>
      </section>

      {/* Live Blockchain Stats */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-20">
        <StatBox
          label="Total Rounds"
          value={totalRounds.toString()}
          icon="analytics"
        />
        <StatBox
          label="Total Rewards Pool"
          value={`${totalPool.toLocaleString()} ${tokenSymbol || "SORE"}`}
          color="text-primary"
          icon="token"
        />
        <StatBox
          label="System Status"
          value={activeRound ? "Voting Open" : "Standby"}
          color={activeRound ? "text-green-500" : "text-slate-500"}
          icon={activeRound ? "check_circle" : "pause_circle"}
        />
      </section>
    </div>
  );
}

// Reusable Stat Component
const StatBox = ({ label, value, color = "text-white", icon }: any) => (
  <div className="p-8 bg-surface-dark border border-border-dark rounded-2xl text-center hover:border-primary/30 transition group">
    <div className="flex justify-center mb-3 text-slate-600 group-hover:text-primary transition-colors">
      {icon && (
        <span className="material-symbols-outlined text-3xl">{icon}</span>
      )}
    </div>
    <p className="text-slate-500 uppercase text-xs font-bold tracking-widest mb-2">
      {label}
    </p>
    <p className={`text-3xl font-black ${color}`}>{value}</p>
  </div>
);
