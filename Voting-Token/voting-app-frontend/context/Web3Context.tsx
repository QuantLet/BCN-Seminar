"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
} from "react";
import { ethers, Contract } from "ethers";
import { Round, Candidate, RewardRecord } from "../app/types";

// --- Config ---
const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_VOTING_ADDRESS || "";

const VOTING_ABI = [
  "function owner() view returns (address)",
  "function rewardToken() view returns (address)",
  "function roundCount() view returns (uint256)",
  "function rounds(uint256) view returns (uint256 id, string title, uint256 endTime, bool isActive, uint256 totalRewardPool)",
  "function getCandidates(uint256) view returns (tuple(uint256 id, string name, uint256 voteCount, address wallet, address[] voters)[])",
  "function hasVoted(uint256, address) view returns (bool)",
  "function usernames(address) view returns (string)",
  "function setUsername(string)",
  "function getUserRewardHistory(address) view returns (tuple(uint256 roundId, string roundTitle, uint256 amount, uint256 rank, uint256 timestamp)[])",
  "function vote(uint256, uint256)",
  "function createRound(string, uint256, uint256)",
  "function addCandidate(uint256, string, address)",
  "function endRound(uint256)",
];

const TOKEN_ABI = [
  "function balanceOf(address) view returns (uint256)",
  "function mint(address, uint256)",
  "function symbol() view returns (string)",
];

interface Web3ContextType {
  account: string | null;
  username: string;
  isAdmin: boolean;
  isConnected: boolean;

  rounds: Round[];
  activeRound: Round | null; // <--- ADDED BACK (Computed)

  allCandidates: Record<number, Candidate[]>;
  userVoteStatus: Record<number, boolean>;

  userRewards: RewardRecord[];
  contract: Contract | null;
  contractBalance: string;
  tokenSymbol: string;
  loading: boolean;

  connectWallet: () => Promise<void>;
  disconnectWallet: () => void;
  refreshData: () => Promise<void>;
  registerUser: () => Promise<void>;
  refillTreasury: (amount: string) => Promise<void>;
  addTokenToMetaMask: () => Promise<void>;
}

const Web3Context = createContext<Web3ContextType | undefined>(undefined);

export const Web3Provider = ({ children }: { children: React.ReactNode }) => {
  const [account, setAccount] = useState<string | null>(null);
  const [contract, setContract] = useState<Contract | null>(null);
  const [tokenContract, setTokenContract] = useState<Contract | null>(null);

  const [username, setUsername] = useState<string>("");
  const [isAdmin, setIsAdmin] = useState(false);
  const [contractBalance, setContractBalance] = useState("0");
  const [tokenSymbol, setTokenSymbol] = useState("");

  // Data States
  const [rounds, setRounds] = useState<Round[]>([]);
  const [allCandidates, setAllCandidates] = useState<
    Record<number, Candidate[]>
  >({});
  const [userVoteStatus, setUserVoteStatus] = useState<Record<number, boolean>>(
    {},
  );
  const [userRewards, setUserRewards] = useState<RewardRecord[]>([]);
  const [loading, setLoading] = useState(false);

  // --- Computed Property: Active Round ---
  // Finds the most recent round where isActive is true
  const activeRound = useMemo(() => {
    return rounds.find((r) => r.isActive) || null;
  }, [rounds]);

  const connectWallet = async () => {
    if (typeof window !== "undefined" && window.ethereum) {
      try {
        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        const address = await signer.getAddress();
        setAccount(address);

        const _contract = new ethers.Contract(
          CONTRACT_ADDRESS,
          VOTING_ABI,
          signer,
        );
        setContract(_contract);

        try {
          const owner = await _contract.owner();
          setIsAdmin(owner.toLowerCase() === address.toLowerCase());
        } catch (e) {
          console.error("Owner check failed", e);
        }

        await checkUser(_contract, address);
        await loadInitialData(_contract, signer, address);
      } catch (err) {
        console.error("Connection Error:", err);
      }
    } else {
      alert("Please install MetaMask!");
    }
  };

  // ADD THIS FUNCTION
  const addTokenToMetaMask = async () => {
    if (!tokenContract || !tokenSymbol) return;

    try {
      const tokenAddress = await tokenContract.getAddress();

      // This is the MetaMask API to register a token
      await window.ethereum.request({
        method: "wallet_watchAsset",
        params: {
          type: "ERC20",
          options: {
            address: tokenAddress,
            symbol: tokenSymbol,
            decimals: 18,
            image: "https://i.postimg.cc/bJJvKVBT/logo_new.png", // Optional: You can put a URL to your logo here
          },
        },
      });
    } catch (error) {
      console.error(error);
    }
  };

  const disconnectWallet = () => {
    setAccount(null);
    setContract(null);
    setTokenContract(null);
    setUsername("");
    setRounds([]);
    setAllCandidates({});
  };

  const checkUser = async (contractInstance: Contract, address: string) => {
    try {
      const name = await contractInstance.usernames(address);
      if (name) setUsername(name);
    } catch (e) {}
  };

  const loadInitialData = useCallback(
    async (contractInstance: Contract, signer: any, address: string) => {
      setLoading(true);
      try {
        // 1. Token Data
        try {
          const tokenAddress = await contractInstance.rewardToken();
          const _tokenContract = new ethers.Contract(
            tokenAddress,
            TOKEN_ABI,
            signer,
          );
          setTokenContract(_tokenContract);
          const bal = await _tokenContract.balanceOf(CONTRACT_ADDRESS);
          const sym = await _tokenContract.symbol();
          // const sym = "SORE"; 
          setContractBalance(ethers.formatEther(bal));
          setTokenSymbol(sym);
        } catch (e) {}

        // 2. Fetch Rounds
        const count = await contractInstance.roundCount();
        const totalRounds = Number(count);

        const loadedRounds: Round[] = [];
        const candidatesMap: Record<number, Candidate[]> = {};
        const voteStatusMap: Record<number, boolean> = {};

        for (let i = totalRounds; i >= 1; i--) {
          const r = await contractInstance.rounds(i);
          const roundId = Number(r.id);

          loadedRounds.push({
            id: roundId,
            title: r.title,
            endTime: Number(r.endTime),
            isActive: r.isActive,
            totalRewardPool: ethers.formatEther(r.totalRewardPool),
          });

          const cands = await contractInstance.getCandidates(roundId);
          candidatesMap[roundId] = cands.map((c: any) => ({
            id: Number(c.id),
            name: c.name,
            voteCount: Number(c.voteCount),
            wallet: c.wallet,
            voters: c.voters,
          }));

          if (address) {
            const hasVoted = await contractInstance.hasVoted(roundId, address);
            voteStatusMap[roundId] = hasVoted;
          }
        }

        setRounds(loadedRounds);
        setAllCandidates(candidatesMap);
        setUserVoteStatus(voteStatusMap);

        // 3. User History
        if (address) {
          const rewards = await contractInstance.getUserRewardHistory(address);
          setUserRewards(
            rewards.map((r: any) => ({
              roundId: Number(r.roundId),
              roundTitle: r.roundTitle,
              amount: ethers.formatEther(r.amount),
              rank: Number(r.rank),
              timestamp: Number(r.timestamp),
            })),
          );
        }
      } catch (e) {
        console.error("Load Data Error:", e);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const refillTreasury = async (amount: string) => {
    if (!tokenContract) return;
    try {
      const amountWei = ethers.parseEther(amount);
      const tx = await tokenContract.mint(CONTRACT_ADDRESS, amountWei);
      await tx.wait();
      const bal = await tokenContract.balanceOf(CONTRACT_ADDRESS);
      setContractBalance(ethers.formatEther(bal));
    } catch (e: any) {
      throw new Error(e.reason || e.message);
    }
  };

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

  return (
    <Web3Context.Provider
      value={{
        account,
        addTokenToMetaMask,
        username,
        isAdmin,
        isConnected: !!account,
        rounds,
        activeRound, // <--- Providing the computed value here
        allCandidates,
        userVoteStatus,
        userRewards,
        contract,
        contractBalance,
        tokenSymbol,
        loading,
        connectWallet,
        disconnectWallet,
        refreshData: () =>
          contract && account && contract.runner
            ? loadInitialData(contract, contract.runner, account)
            : Promise.resolve(),
        registerUser,
        refillTreasury,
      }}
    >
      {children}
    </Web3Context.Provider>
  );
};

export const useWeb3 = () => {
  const context = useContext(Web3Context);
  if (context === undefined)
    throw new Error("useWeb3 must be used within a Web3Provider");
  return context;
};
