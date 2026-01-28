import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import "./globals.css";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import { Web3Provider } from "../context/Web3Context"; // Import Provider
import { UIProvider } from "../context/UIContext";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
  display: "swap",
});

export const metadata: Metadata = {
  title: "QNAR Token Voting DApp",
  description: "Decentralized voting platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className={`${manrope.variable} bg-background-light dark:bg-background-dark text-slate-900 dark:text-white font-display transition-colors duration-300 antialiased flex flex-col min-h-screen`}
      >
        {" "}
        <UIProvider>
          {/* Wrap App in Web3Provider */}
          <Web3Provider>
            <Navbar />
            <main className="flex-grow">{children}</main>
            <Footer />
          </Web3Provider>
        </UIProvider>
      </body>
    </html>
  );
}
