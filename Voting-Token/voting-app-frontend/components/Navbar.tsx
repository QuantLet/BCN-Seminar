"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useWeb3 } from "../context/Web3Context";
import Image from "next/image";

export const Navbar = () => {
  const pathname = usePathname();
  const {
    isConnected,
    account,
    username,
    connectWallet,
    disconnectWallet,
    isAdmin,
  } = useWeb3();

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="sticky top-0 z-50 w-full glass border-b border-border-dark/30 bg-background-dark/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center group cursor-pointer">
          <div className="">
            <Image src="/images/logo.png" alt="Logo" width={80} height={80} />
          </div>
          <h1 className="text-xl font-extrabold tracking-tight text-white">
            QNAR<span className="text-primary"> Voting Token</span>
          </h1>
        </Link>

        {/* --- NAVIGATION LINKS --- */}
        {isConnected && (
          <div className="hidden md:flex items-center gap-8 animate-in fade-in duration-500">
            <NavLink
              href="/explore"
              label="Vote"
              active={isActive("/explore")}
            />
            <NavLink
              href="/leaderboard"
              label="Leaderboard"
              active={isActive("/leaderboard")}
            />

            {/* DYNAMIC LABEL: "Distributions" for Admin, "My Rewards" for Users */}
            <NavLink
              href="/rewards"
              label={isAdmin ? "Distributions" : "My Rewards"}
              active={isActive("/rewards")}
            />

            {isAdmin && (
              <NavLink
                href="/governance"
                label="Admin Dashboard"
                active={
                  isActive("/governance") || isActive("/governance/create")
                }
              />
            )}
          </div>
        )}

        {/* --- CONNECT BUTTON --- */}
        <div className="flex items-center gap-4">
          {!isConnected ? (
            <button
              onClick={connectWallet}
              className="bg-primary hover:bg-primary/90 text-white px-6 py-2.5 rounded-xl font-bold transition-all shadow-lg shadow-primary/20 active:scale-95"
            >
              Connect Wallet
            </button>
          ) : (
            <div className="flex items-center gap-3">
              {isAdmin && (
                <span className="hidden sm:inline-block bg-yellow-500/10 text-yellow-500 text-[10px] font-bold px-2 py-1 rounded border border-yellow-500/20 uppercase tracking-widest">
                  Admin
                </span>
              )}

              <div className="text-right hidden sm:block">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                  Connected
                </p>
                <p className="text-xs font-bold text-primary">
                  {username || `${account?.substring(0, 6)}...`}
                </p>
              </div>
              <button
                onClick={disconnectWallet}
                className="flex items-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 px-4 py-2.5 rounded-xl font-bold transition-all active:scale-95"
              >
                <span className="material-symbols-outlined text-[20px]">
                  logout
                </span>
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

const NavLink = ({
  href,
  label,
  active,
}: {
  href: string;
  label: string;
  active: boolean;
}) => (
  <Link
    href={href}
    className={`text-sm font-bold transition-colors hover:text-primary ${
      active ? "text-primary" : "text-slate-400"
    }`}
  >
    {label}
  </Link>
);
