Refers to "Pointer to Implementation", and is mostly used to clean up headers.

```c++
// ---- Header (would normally live in .hpp) ----
class Impl; // forward declaration of implementation

class Image {
public:
    Image();
    explicit Image(int w, int h);
    ~Image();                    // out-of-line to see full Impl

    void set_pixel(int x, int y, uint8_t v);
    uint8_t get_pixel(int x, int y) const;
    void info() const;

private:
    std::unique_ptr<Impl> p_;    // opaque pointer
};

// ---- Implementation (would normally live in .cpp) ----
class Image::Impl {
public:
    int w{0}, h{0};
    std::vector<uint8_t> data; // flat grayscale

    Impl() = default;
    Impl(int W, int H) : w(W), h(H), data(static_cast<size_t>(W*H), 0) {}

    size_t idx(int x, int y) const { return static_cast<size_t>(y*w + x); }
};

// OR IF ITS AN EXTERNAL LIBRARY, YOU JUST #INCLUDE that library in the .cpp

Image::Image() : p_(std::make_unique<Impl>()) {}
Image::Image(int w, int h) : p_(std::make_unique<Impl>(w,h)) {}
Image::~Image() = default; // needs full Impl in this TU

void Image::set_pixel(int x, int y, uint8_t v) {
    if (x<0 || y<0 || x>=p_->w || y>=p_->h) throw std::out_of_range("pixel");
    p_->data[p_->idx(x,y)] = v;
}
uint8_t Image::get_pixel(int x, int y) const {
    if (x<0 || y<0 || x>=p_->w || y>=p_->h) throw std::out_of_range("pixel");
    return p_->data[p_->idx(x,y)];
}
void Image::info() const { std::cout << "Image " << p_->w << "x" << p_->h << "\n"; }

```