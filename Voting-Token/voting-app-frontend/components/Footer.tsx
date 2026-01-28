import Image from "next/image";
import Link from "next/link";

export const Footer = () => {
  return (
    <footer className="border-t border-border-dark/30 py-12 mt-20">
      <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="rounded-lg flex items-center justify-center">
          <Link
            href="https://quantlet.com/"
            className="ml-2"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="/images/img3.png"
              alt="VOTE.io Logo"
              width={50}
              height={50}
            />
          </Link>
        </div>
        <div className="flex gap-8 text-sm text-slate-500 font-medium">
          {/* <a className="hover:text-white transition-colors" href="#">
            Docs
          </a>
          <a className="hover:text-white transition-colors" href="#">
            Governance Forum
          </a>
          <a className="hover:text-white transition-colors" href="#">
            Discord
          </a>
          <a className="hover:text-white transition-colors" href="#">
            X / Twitter
          </a> */}
        </div>
        <div className="flex justify-center items-center">
          <p className="text-slate-500 text-xs">© 2026 Quantlet. Built for</p>
          <Link
            href="https://quantinar.com/"
            className="ml-2"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/images/img2_new.png"
              alt="QNAR Voting Token"
              width={50}
              height={50}
            />
          </Link>
        </div>
      </div>
    </footer>
  );
};
