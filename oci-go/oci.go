package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"golang.org/x/term"
)

// OCIError represents an error that occurred during OCI operations
type OCIError struct {
	Code    int
	Message string
}

// OCIResult represents the result of pulling a model
type OCIResult struct {
	LocalPath string
	Digest    string
	Error     *OCIError
}

//export PullOCIModel
func PullOCIModel(imageRef, cacheDir *C.char) *C.char {
	goImageRef := C.GoString(imageRef)
	goCacheDir := C.GoString(cacheDir)

	result, err := pullModel(goImageRef, goCacheDir)
	if err != nil {
		result = &OCIResult{
			Error: &OCIError{
				Code:    1,
				Message: err.Error(),
			},
		}
	}

	jsonBytes, _ := json.Marshal(result)
	return C.CString(string(jsonBytes))
}

//export FreeString
func FreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}

// progressWriter wraps an io.Writer and tracks progress with docker-style output
type progressWriter struct {
	writer      io.Writer
	total       int64
	downloaded  int64
	lastPrint   time.Time
	layerDigest string
	startTime   time.Time
}

func newProgressWriter(w io.Writer, total int64, digest string) *progressWriter {
	return &progressWriter{
		writer:      w,
		total:       total,
		downloaded:  0,
		lastPrint:   time.Now(),
		layerDigest: digest,
		startTime:   time.Now(),
	}
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.writer.Write(p)
	if n > 0 {
		atomic.AddInt64(&pw.downloaded, int64(n))

		// Update progress display every 100ms
		now := time.Now()
		if now.Sub(pw.lastPrint) >= 100*time.Millisecond {
			pw.printProgress()
			pw.lastPrint = now
		}
	}
	return n, err
}

func (pw *progressWriter) printProgress() {
	downloaded := atomic.LoadInt64(&pw.downloaded)

	// Calculate percentage and download speed
	var percentage float64
	if pw.total > 0 {
		percentage = float64(downloaded) / float64(pw.total) * 100.0
	}

	elapsed := time.Since(pw.startTime).Seconds()
	speed := float64(0)
	if elapsed > 0 {
		speed = float64(downloaded) / elapsed / (1024 * 1024) // MB/s
	}

	// Format sizes
	downloadedMB := float64(downloaded) / (1024 * 1024)
	totalMB := float64(pw.total) / (1024 * 1024)

	// Get short digest (first 12 chars after sha256:)
	shortDigest := pw.layerDigest
	if strings.HasPrefix(shortDigest, "sha256:") {
		shortDigest = shortDigest[7:19]
	}

	// Get terminal width, default to 80 if cannot be determined
	termWidth := 80
	if width, _, err := term.GetSize(int(os.Stderr.Fd())); err == nil && width > 0 {
		termWidth = width
	}

	// Print docker-style progress
	if pw.total > 0 {
		// Build the progress message to measure its length
		// Format: "shortDigest: Downloading [] 100.0% (9999.99 MB / 9999.99 MB) 999.99 MB/s"
		prefix := fmt.Sprintf("%s: Downloading [", shortDigest)
		suffix := fmt.Sprintf("] %.1f%% (%.2f MB / %.2f MB) %.2f MB/s",
			percentage, downloadedMB, totalMB, speed)

		// Calculate available space for progress bar
		// Reserve at least 10 chars for the bar, use remaining space up to 50 chars
		fixedWidth := len(prefix) + len(suffix)
		maxBarWidth := termWidth - fixedWidth
		if maxBarWidth < 10 {
			maxBarWidth = 10
		} else if maxBarWidth > 50 {
			maxBarWidth = 50
		}

		// Build the complete line
		var line strings.Builder
		line.WriteString("\r")
		line.WriteString(prefix)

		// Progress bar
		filled := int(float64(maxBarWidth) * percentage / 100.0)
		for i := 0; i < maxBarWidth; i++ {
			if i < filled {
				line.WriteString("=")
			} else if i == filled {
				line.WriteString(">")
			} else {
				line.WriteString(" ")
			}
		}

		line.WriteString(suffix)

		// Pad with spaces to clear any trailing characters from previous output
		currentLen := len(prefix) + maxBarWidth + len(suffix)
		if currentLen < termWidth {
			padding := termWidth - currentLen
			for i := 0; i < padding; i++ {
				line.WriteString(" ")
			}
		}

		// Write the complete line
		fmt.Fprint(os.Stderr, line.String())
	} else {
		// Build line for unknown total size
		line := fmt.Sprintf("\r%s: Downloading %.2f MB %.2f MB/s",
			shortDigest, downloadedMB, speed)

		// Pad with spaces to clear trailing characters
		if len(line) < termWidth {
			padding := termWidth - len(line)
			for i := 0; i < padding; i++ {
				line += " "
			}
		}

		fmt.Fprint(os.Stderr, line)
	}
}

func (pw *progressWriter) finish() {
	downloaded := atomic.LoadInt64(&pw.downloaded)
	downloadedMB := float64(downloaded) / (1024 * 1024)

	// Get short digest
	shortDigest := pw.layerDigest
	if strings.HasPrefix(shortDigest, "sha256:") {
		shortDigest = shortDigest[7:19]
	}

	fmt.Fprintf(os.Stderr, "\r%s: Download complete (%.2f MB)\n", shortDigest, downloadedMB)
}

func pullModel(imageRef, cacheDir string) (*OCIResult, error) {
	ctx := context.Background()

	// Parse the image reference
	ref, err := name.ParseReference(imageRef)
	if err != nil {
		return nil, fmt.Errorf("failed to parse image reference: %w", err)
	}

	// Use docker config for authentication (supports docker login)
	authenticator := authn.NewMultiKeychain(
		authn.DefaultKeychain,
	)

	// Get the image descriptor
	img, err := remote.Image(ref, remote.WithAuthFromKeychain(authenticator), remote.WithContext(ctx))
	if err != nil {
		return nil, fmt.Errorf("failed to fetch image: %w", err)
	}

	// Get the manifest
	manifest, err := img.Manifest()
	if err != nil {
		return nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	// Find the GGUF layer
	var ggufLayer v1.Layer
	var ggufDigest string
	var layerSize int64
	for _, layer := range manifest.Layers {
		mediaType := string(layer.MediaType)
		if mediaType == "application/vnd.docker.ai.gguf.v3" || strings.Contains(mediaType, "gguf") {
			ggufLayer, err = img.LayerByDigest(layer.Digest)
			if err != nil {
				return nil, fmt.Errorf("failed to get GGUF layer: %w", err)
			}
			ggufDigest = layer.Digest.String()
			layerSize = layer.Size
			break
		}
	}

	if ggufLayer == nil {
		return nil, fmt.Errorf("no GGUF layer found in image")
	}

	// Prepare local file path
	refStr := ref.String()
	modelFilename := strings.ReplaceAll(refStr, "/", "_")
	modelFilename = strings.ReplaceAll(modelFilename, ":", "_")
	modelFilename += ".gguf"

	localPath := filepath.Join(cacheDir, modelFilename)
	tempPath := localPath + ".tmp"
	digestPath := localPath + ".digest"

	// Check if file already exists and is complete
	if _, err := os.Stat(localPath); err == nil {
		// File exists, verify digest matches
		if storedDigest, err := os.ReadFile(digestPath); err == nil && string(storedDigest) == ggufDigest {
			fmt.Fprintf(os.Stderr, "%s: Using cached model (digest verified)\n", ggufDigest[7:19])
			return &OCIResult{
				LocalPath: localPath,
				Digest:    ggufDigest,
			}, nil
		}
		// Digest mismatch or missing, need to re-download
		fmt.Fprintf(os.Stderr, "%s: Cache digest mismatch, re-downloading\n", ggufDigest[7:19])
		os.Remove(localPath)
		os.Remove(digestPath)
		os.Remove(tempPath)
	}

	// Check for partial download
	var existingSize int64 = 0
	var resuming bool = false
	if fileInfo, err := os.Stat(tempPath); err == nil {
		// Verify the digest matches what we expect
		if storedDigest, err := os.ReadFile(digestPath); err == nil && string(storedDigest) == ggufDigest {
			existingSize = fileInfo.Size()
			if existingSize > 0 && existingSize < layerSize {
				resuming = true
			}
		} else {
			// Digest mismatch, remove partial file
			os.Remove(tempPath)
			os.Remove(digestPath)
		}
	}

	// Store digest for verification
	if err := os.WriteFile(digestPath, []byte(ggufDigest), 0644); err != nil {
		return nil, fmt.Errorf("failed to write digest file: %w", err)
	}

	// Download the layer
	layerReader, err := ggufLayer.Uncompressed()
	if err != nil {
		return nil, fmt.Errorf("failed to get layer reader: %w", err)
	}
	defer layerReader.Close()

	// Skip already downloaded bytes if resuming
	if resuming && existingSize > 0 {
		_, err = io.CopyN(io.Discard, layerReader, existingSize)
		if err != nil {
			return nil, fmt.Errorf("failed to skip downloaded bytes: %w", err)
		}
	}

	// Open file for appending or create new
	var outFile *os.File
	if resuming {
		outFile, err = os.OpenFile(tempPath, os.O_APPEND|os.O_WRONLY, 0644)
	} else {
		outFile, err = os.Create(tempPath)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to create output file: %w", err)
	}

	// Create progress writer
	pw := newProgressWriter(outFile, layerSize, ggufDigest)
	if resuming {
		atomic.StoreInt64(&pw.downloaded, existingSize)
		pw.startTime = time.Now() // Reset start time for accurate speed calculation
	}

	// Copy the data with progress tracking
	_, err = io.Copy(pw, layerReader)
	outFile.Close()

	if err != nil {
		// Don't remove partial file on error - allow resume
		return nil, fmt.Errorf("failed to write layer data: %w", err)
	}

	// Print completion message
	pw.finish()

	// Verify downloaded file size matches expected
	if fileInfo, err := os.Stat(tempPath); err == nil {
		if fileInfo.Size() != layerSize {
			return nil, fmt.Errorf("downloaded file size (%d) doesn't match expected size (%d)",
				fileInfo.Size(), layerSize)
		}
	}

	// Rename to final location (atomic operation)
	if err := os.Rename(tempPath, localPath); err != nil {
		return nil, fmt.Errorf("failed to rename file: %w", err)
	}

	return &OCIResult{
		LocalPath: localPath,
		Digest:    ggufDigest,
	}, nil
}

func main() {}
