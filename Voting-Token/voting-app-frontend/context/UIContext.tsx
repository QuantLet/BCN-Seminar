"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
} from "react";

// --- Types ---
type ToastType = "success" | "error" | "info";

interface UIContextType {
  showLoader: (message?: string) => void;
  hideLoader: () => void;
  showToast: (message: string, type?: ToastType) => void;
  confirmAction: (message: string) => Promise<boolean>;
}

const UIContext = createContext<UIContextType | undefined>(undefined);

export const UIProvider = ({ children }: { children: React.ReactNode }) => {
  // --- LOADER STATE ---
  const [isLoading, setIsLoading] = useState(false);
  const [loaderMessage, setLoaderMessage] = useState(
    "Processing Transaction..."
  );

  // --- TOAST STATE ---
  const [toast, setToast] = useState<{ msg: string; type: ToastType } | null>(
    null
  );

  // --- CONFIRM MODAL STATE ---
  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean;
    message: string;
    resolve: (val: boolean) => void;
  }>({ isOpen: false, message: "", resolve: () => {} });

  // 1. LOADER FUNCTIONS
  const showLoader = (msg = "Processing...") => {
    setLoaderMessage(msg);
    setIsLoading(true);
  };
  const hideLoader = () => setIsLoading(false);

  // 2. TOAST FUNCTION
  const showToast = (message: string, type: ToastType = "info") => {
    setToast({ msg: message, type });
    setTimeout(() => setToast(null), 4000); // Auto hide after 4s
  };

  // 3. CONFIRMATION FUNCTION (Returns a Promise!)
  const confirmAction = useCallback((message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmModal({
        isOpen: true,
        message,
        resolve: (val: boolean) => {
          setConfirmModal((prev) => ({ ...prev, isOpen: false }));
          resolve(val);
        },
      });
    });
  }, []);

  return (
    <UIContext.Provider
      value={{ showLoader, hideLoader, showToast, confirmAction }}
    >
      {children}

      {/* --- RENDER LOADER --- */}
      {isLoading && (
        <div className="fixed inset-0 z-[60] flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="relative size-20">
            <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
            <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
          </div>
          <p className="mt-6 text-xl font-bold text-white animate-pulse">
            {loaderMessage}
          </p>
          <p className="text-sm text-slate-400 mt-2">
            Check MetaMask for approval
          </p>
        </div>
      )}

      {/* --- RENDER TOAST --- */}
      {toast && (
        <div
          className={`fixed top-6 right-6 z-[70] px-6 py-4 rounded-xl shadow-2xl border flex items-center gap-3 animate-in slide-in-from-right duration-300 ${
            toast.type === "success"
              ? "bg-emerald-900/90 border-emerald-500 text-white"
              : toast.type === "error"
              ? "bg-red-900/90 border-red-500 text-white"
              : "bg-slate-800 border-slate-600 text-white"
          }`}
        >
          <span className="material-symbols-outlined text-xl">
            {toast.type === "success"
              ? "check_circle"
              : toast.type === "error"
              ? "error"
              : "info"}
          </span>
          <p className="font-bold text-sm">{toast.msg}</p>
        </div>
      )}

      {/* --- RENDER CONFIRM MODAL --- */}
      {confirmModal.isOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-surface-dark border border-border-dark p-8 rounded-2xl max-w-md w-full shadow-2xl transform scale-100 animate-in zoom-in-95 duration-200">
            <h3 className="text-xl font-bold mb-2 text-white">
              Confirmation Required
            </h3>
            <p className="text-slate-400 mb-8">{confirmModal.message}</p>
            <div className="flex gap-4">
              <button
                onClick={() => confirmModal.resolve(false)}
                className="flex-1 px-4 py-3 rounded-xl border border-border-dark hover:bg-white/5 font-bold transition"
              >
                Cancel
              </button>
              <button
                onClick={() => confirmModal.resolve(true)}
                className="flex-1 px-4 py-3 rounded-xl bg-primary hover:bg-primary/90 text-white font-bold transition shadow-lg shadow-primary/20"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}
    </UIContext.Provider>
  );
};

export const useUI = () => {
  const context = useContext(UIContext);
  if (!context) throw new Error("useUI must be used within UIProvider");
  return context;
};
