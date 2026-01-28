// types.ts
export interface Candidate {
  id: number;
  name: string;
  voteCount: number;
  wallet: string;
  voters?: string[];
}

export interface Round {
  id: number;
  title: string;
  endTime: number;
  isActive: boolean;
  totalRewardPool: string;
}

export interface RewardRecord {
  roundId: number;
  roundTitle: string;
  amount: string;
  rank: number;
  timestamp: number;
}

export interface UserStats {
  username: string;
  totalEarned: string;
  history: RewardRecord[];
}
