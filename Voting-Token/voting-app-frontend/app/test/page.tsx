"use client";
import { useState, useEffect } from "react";
import { ethers, Contract, BrowserProvider, JsonRpcSigner } from "ethers";

// 1. Fix "Property 'ethereum' does not exist on type 'Window'"
declare global {
  interface Window {
    ethereum: any;
  }
}

const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_VOTING_ADDRESS || "";

// 2. Define Types for your Data
interface Candidate {
  id: number;
  name: string;
  voteCount: number;
  wallet: string;
}

interface Round {
  id: number;
  title: string;
  endTime: number;
  isActive: boolean;
  totalRewardPool: string; // Keep as string for display (formatted ether)
}

// 3. Minimal ABI
const VOTING_ABI = [
  "function roundCount() view returns (uint256)",
  "function rounds(uint256) view returns (uint256 id, string title, uint256 endTime, bool isActive, uint256 totalRewardPool)",
  "function getCandidates(uint256) view returns (tuple(uint256 id, string name, uint256 voteCount, address wallet)[])",
  "function usernames(address) view returns (string)",
  "function setUsername(string)",
  "function vote(uint256, uint256)",
  "function createRound(string, uint256, uint256)",
  "function addCandidate(uint256, string, address)",
  "function endRound(uint256)"
];

export default function Home() {
  // --- State Definitions with Types ---
  const [provider, setProvider] = useState<BrowserProvider | null>(null);
  const [signer, setSigner] = useState<JsonRpcSigner | null>(null);
  const [contract, setContract] = useState<Contract | null>(null);
  const [account, setAccount] = useState<string>("");
  const [username, setUsername] = useState<string>("");

  const [activeRound, setActiveRound] = useState<Round | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  // Admin Inputs
  const [newRoundTitle, setNewRoundTitle] = useState("");
  const [candName, setCandName] = useState("");
  const [candAddress, setCandAddress] = useState("");

  // --- Wallet Connection ---
  const connectWallet = async () => {
    if (typeof window !== "undefined" && window.ethereum) {
      try {
        const _provider = new ethers.BrowserProvider(window.ethereum);
        const _signer = await _provider.getSigner();
        setProvider(_provider);
        setSigner(_signer);

        const address = await _signer.getAddress();
        setAccount(address);

        // Initialize Contract
        const _contract = new ethers.Contract(CONTRACT_ADDRESS, VOTING_ABI, _signer);
        setContract(_contract);

        // Fetch Initial Data
        checkUser(_contract, address);
        loadData(_contract);
      } catch (err) {
        console.error("Connection Error:", err);
      }
    } else {
      alert("Please install MetaMask!");
    }
  };

  // --- Fetch Data Helper ---
  const checkUser = async (contractInstance: Contract, address: string) => {
    try {
      const name = await contractInstance.usernames(address);
      if (name) setUsername(name);
    } catch (e) {
      console.log("No username found.");
    }
  };

  const loadData = async (contractInstance: Contract) => {
    try {
      const count = await contractInstance.roundCount(); // Returns BigInt
      
      if (Number(count) > 0) {
        const roundData = await contractInstance.rounds(count);
        
        // Map Solidity Struct to TS Interface
        const formattedRound: Round = {
          id: Number(roundData.id),
          title: roundData.title,
          endTime: Number(roundData.endTime),
          isActive: roundData.isActive,
          totalRewardPool: ethers.formatEther(roundData.totalRewardPool)
        };
        
        setActiveRound(formattedRound);

        const candsData = await contractInstance.getCandidates(count);
        
        // Map Candidate Array
        // Note: 'c' is an array-like object in Ethers v6 result
        const formattedCands: Candidate[] = candsData.map((c: any) => ({
          id: Number(c.id), // or c[0]
          name: c.name,     // or c[1]
          voteCount: Number(c.voteCount), // or c[2]
          wallet: c.wallet  // or c[3]
        }));
        
        setCandidates(formattedCands);
      }
    } catch (e) {
      console.error("Error loading round data:", e);
    }
  };

  // --- User Actions ---
  const registerUser = async () => {
    const name = prompt("Enter username:");
    if (!name || !contract) return;
    try {
      const tx = await contract.setUsername(name);
      await tx.wait();
      setUsername(name);
    } catch (err) {
      alert("Registration failed");
    }
  };

  const handleVote = async (candidateId: number) => {
    if (!contract || !activeRound) return;
    try {
      setLoading(true);
      const tx = await contract.vote(activeRound.id, candidateId);
      await tx.wait();
      alert("Vote Casted Successfully!");
      loadData(contract); // Refresh data
    } catch (err: any) {
      // Ethers v6 error parsing
      const errorMessage = err.reason || err.message || "Transaction failed";
      alert("Vote Failed: " + errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // --- Admin Actions ---
  const createRound = async () => {
    if (!contract) return;
    try {
      const duration = 3600; // 1 hour
      const reward = ethers.parseEther("10"); // 10 Tokens
      const tx = await contract.createRound(newRoundTitle, duration, reward);
      await tx.wait();
      alert("Round Created!");
      loadData(contract);
    } catch (e) {
      console.error(e);
      alert("Failed to create round. Check console.");
    }
  };

  const addCandidateToRound = async () => {
    if (!contract || !activeRound) return;
    try {
      const tx = await contract.addCandidate(activeRound.id, candName, candAddress);
      await tx.wait();
      alert("Candidate Added!");
      loadData(contract);
    } catch (e) {
      console.error(e);
      alert("Failed to add candidate.");
    }
  };

  const endCurrentRound = async () => {
    if (!contract || !activeRound) return;
    try {
      const tx = await contract.endRound(activeRound.id);
      await tx.wait();
      alert("Round Ended & Rewards Distributed!");
      loadData(contract);
    } catch (e) {
      console.error(e);
      alert("Error ending round (Check balance or status).");
    }
  };

  // --- Render ---
  return (
    <main className="min-h-screen bg-slate-900 text-white font-sans p-6">
      {/* Navbar */}
      <nav className="flex justify-between items-center mb-10 border-b border-slate-700 pb-4">
        <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
          DecenVote
        </h1>
        {!account ? (
          <button
            onClick={connectWallet}
            className="bg-blue-600 hover:bg-blue-700 px-5 py-2 rounded-lg font-medium transition"
          >
            Connect Wallet
          </button>
        ) : (
          <div className="text-right">
            <p className="text-xs text-slate-400 mb-1">
              {account.substring(0, 6)}...{account.substring(account.length - 4)}
            </p>
            <p className="font-bold text-green-400 cursor-pointer hover:text-green-300">
              {username || (
                <span onClick={registerUser} className="underline decoration-dotted">
                  Set Username
                </span>
              )}
            </p>
          </div>
        )}
      </nav>

      {/* Admin Panel (Conditional) */}
      {contract && (
        <div className="mb-10 p-6 bg-slate-800 rounded-xl border border-slate-700 shadow-xl">
          <h2 className="text-xl font-bold text-yellow-500 mb-4 flex items-center gap-2">
            <span>⚡</span> Admin Zone
          </h2>
          
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <input
              type="text"
              placeholder="Round Title (e.g. Best Dev Tool)"
              className="bg-slate-900 border border-slate-600 rounded px-4 py-2 text-white focus:outline-none focus:border-blue-500 flex-grow"
              onChange={(e) => setNewRoundTitle(e.target.value)}
            />
            <button
              onClick={createRound}
              className="bg-yellow-600 hover:bg-yellow-500 px-6 py-2 rounded font-semibold transition"
            >
              Start Round
            </button>
            <button
              onClick={endCurrentRound}
              className="bg-red-600 hover:bg-red-500 px-6 py-2 rounded font-semibold transition"
            >
              End Round
            </button>
          </div>

          <div className="flex flex-col md:flex-row gap-4 items-center bg-slate-900 p-4 rounded-lg">
            <span className="text-sm font-bold text-slate-400">ADD CANDIDATE:</span>
            <input
              placeholder="Name"
              className="bg-slate-800 border border-slate-600 rounded px-3 py-1 text-white"
              onChange={(e) => setCandName(e.target.value)}
            />
            <input
              placeholder="Wallet (0x...)"
              className="bg-slate-800 border border-slate-600 rounded px-3 py-1 text-white flex-grow font-mono text-sm"
              onChange={(e) => setCandAddress(e.target.value)}
            />
            <button
              onClick={addCandidateToRound}
              className="bg-green-600 hover:bg-green-500 px-4 py-1 rounded font-bold text-lg"
            >
              +
            </button>
          </div>
        </div>
      )}

      {/* Active Round UI */}
      {activeRound ? (
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-8">
            <div>
              <h2 className="text-5xl font-bold mb-2">{activeRound.title}</h2>
              <div className="flex items-center gap-3">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
                    activeRound.isActive ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                  }`}
                >
                  {activeRound.isActive ? "● Live Now" : "● Ended"}
                </span>
                <span className="text-slate-500 text-sm">
                  Ends: {new Date(activeRound.endTime * 1000).toLocaleTimeString()}
                </span>
              </div>
            </div>
            <div className="mt-4 md:mt-0 bg-gradient-to-r from-purple-900 to-indigo-900 p-4 rounded-xl border border-purple-500/30">
              <p className="text-xs text-purple-300 uppercase tracking-wider mb-1">Reward Pool</p>
              <p className="text-3xl font-mono font-bold text-white">
                {activeRound.totalRewardPool} <span className="text-sm">TOKENS</span>
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {candidates.map((cand) => (
              <div
                key={cand.id}
                className="group bg-slate-800 p-6 rounded-2xl border border-slate-700 hover:border-blue-500/50 transition-all shadow-lg hover:shadow-blue-500/10 flex justify-between items-center"
              >
                <div>
                  <h3 className="text-2xl font-bold text-white group-hover:text-blue-400 transition">
                    {cand.name}
                  </h3>
                  <p className="text-slate-500 text-xs font-mono mb-3">
                    {cand.wallet.slice(0, 6)}...{cand.wallet.slice(-6)}
                  </p>
                  <div className="inline-flex items-center gap-2 bg-slate-900 px-3 py-1 rounded-lg border border-slate-700">
                    <span className="text-slate-400 text-xs uppercase">Current Votes</span>
                    <span className="text-white font-mono font-bold text-lg">{cand.voteCount}</span>
                  </div>
                </div>

                {activeRound.isActive && (
                  <button
                    onClick={() => handleVote(cand.id)}
                    disabled={loading}
                    className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 text-white px-6 py-4 rounded-xl font-bold transition-all transform hover:scale-105 active:scale-95 shadow-lg shadow-blue-900/20"
                  >
                    {loading ? (
                      <span className="animate-pulse">Voting...</span>
                    ) : (
                      "VOTE"
                    )}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-64 bg-slate-800/50 rounded-2xl border border-slate-700 border-dashed">
          <p className="text-xl text-slate-400">No active voting events found.</p>
          <p className="text-slate-500 text-sm mt-2">Connect as Admin to start one.</p>
        </div>
      )}
    </main>
  );
}